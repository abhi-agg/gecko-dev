/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "ScriptLoader.h"
#include "ModuleLoader.h"

#include "jsapi.h"
#include "js/CompileOptions.h"  // JS::CompileOptions, JS::InstantiateOptions
#include "js/ContextOptions.h"  // JS::ContextOptionsRef
#include "js/experimental/JSStencil.h"  // JS::Stencil, JS::CompileModuleScriptToStencil, JS::InstantiateModuleStencil
#include "js/MemoryFunctions.h"
#include "js/Modules.h"  // JS::FinishDynamicModuleImport, JS::{G,S}etModuleResolveHook, JS::Get{ModulePrivate,ModuleScript,RequestedModule{s,Specifier,SourcePos}}, JS::SetModule{DynamicImport,Metadata}Hook
#include "js/OffThreadScriptCompilation.h"
#include "js/PropertyAndElement.h"  // JS_DefineProperty
#include "js/Realm.h"
#include "js/SourceText.h"
#include "js/loader/LoadedScript.h"
#include "js/loader/ScriptLoadRequest.h"
#include "js/loader/ModuleLoaderBase.h"
#include "js/loader/ModuleLoadRequest.h"
#include "xpcpublic.h"
#include "GeckoProfiler.h"
#include "nsIContent.h"
#include "nsJSUtils.h"
#include "mozilla/dom/AutoEntryScript.h"
#include "mozilla/dom/Document.h"
#include "mozilla/dom/Element.h"
#include "nsGlobalWindowInner.h"
#include "nsIPrincipal.h"
#include "mozilla/LoadInfo.h"

using JS::SourceText;
using namespace JS::loader;

namespace mozilla::dom {

#undef LOG
#define LOG(args) \
  MOZ_LOG(ScriptLoader::gScriptLoaderLog, mozilla::LogLevel::Debug, args)

#define LOG_ENABLED() \
  MOZ_LOG_TEST(ScriptLoader::gScriptLoaderLog, mozilla::LogLevel::Debug)

//////////////////////////////////////////////////////////////
// DOM module loader
//////////////////////////////////////////////////////////////

ModuleLoader::ModuleLoader(ScriptLoader* aLoader) : ModuleLoaderBase(aLoader) {}

ScriptLoader* ModuleLoader::GetScriptLoader() {
  return static_cast<ScriptLoader*>(mLoader.get());
}

bool ModuleLoader::CanStartLoad(ModuleLoadRequest* aRequest, nsresult* aRvOut) {
  if (!GetScriptLoader()->GetDocument()) {
    *aRvOut = NS_ERROR_NULL_POINTER;
    return false;
  }

  // If this document is sandboxed without 'allow-scripts', abort.
  if (GetScriptLoader()->GetDocument()->HasScriptsBlockedBySandbox()) {
    *aRvOut = NS_OK;
    return false;
  }

  // To prevent dynamic code execution, content scripts can only
  // load moz-extension URLs.
  nsCOMPtr<nsIPrincipal> principal = aRequest->TriggeringPrincipal();
  if (BasePrincipal::Cast(principal)->ContentScriptAddonPolicy() &&
      !aRequest->mURI->SchemeIs("moz-extension")) {
    *aRvOut = NS_ERROR_DOM_WEBEXT_CONTENT_SCRIPT_URI;
    return false;
  }

  if (LOG_ENABLED()) {
    nsAutoCString url;
    aRequest->mURI->GetAsciiSpec(url);
    LOG(("ScriptLoadRequest (%p): Start Module Load (url = %s)", aRequest,
         url.get()));
  }

  return true;
}

nsresult ModuleLoader::StartFetch(ModuleLoadRequest* aRequest) {
  nsSecurityFlags securityFlags;

  // According to the spec, module scripts have different behaviour to classic
  // scripts and always use CORS. Only exception: Non linkable about: pages
  // which load local module scripts.
  if (GetScriptLoader()->IsAboutPageLoadingChromeURI(
          aRequest, GetScriptLoader()->GetDocument())) {
    securityFlags = nsILoadInfo::SEC_ALLOW_CROSS_ORIGIN_SEC_CONTEXT_IS_NULL;
  } else {
    securityFlags = nsILoadInfo::SEC_REQUIRE_CORS_INHERITS_SEC_CONTEXT;
    if (aRequest->CORSMode() == CORS_NONE ||
        aRequest->CORSMode() == CORS_ANONYMOUS) {
      securityFlags |= nsILoadInfo::SEC_COOKIES_SAME_ORIGIN;
    } else {
      MOZ_ASSERT(aRequest->CORSMode() == CORS_USE_CREDENTIALS);
      securityFlags |= nsILoadInfo::SEC_COOKIES_INCLUDE;
    }
  }

  securityFlags |= nsILoadInfo::SEC_ALLOW_CHROME;

  // Delegate Shared Behavior to base ScriptLoader
  nsresult rv = GetScriptLoader()->StartLoadInternal(aRequest, securityFlags);
  NS_ENSURE_SUCCESS(rv, rv);

  LOG(("ScriptLoadRequest (%p): Start fetching module", aRequest));

  return NS_OK;
}

void ModuleLoader::ProcessLoadedModuleTree(ModuleLoadRequest* aRequest) {
  MOZ_ASSERT(aRequest->IsReadyToRun());

  if (aRequest->IsTopLevel()) {
    if (aRequest->IsDynamicImport() ||
        (aRequest->GetLoadContext()->mIsInline &&
         aRequest->GetLoadContext()->GetParserCreated() == NOT_FROM_PARSER)) {
      GetScriptLoader()->RunScriptWhenSafe(aRequest);
    } else {
      GetScriptLoader()->MaybeMoveToLoadedList(aRequest);
      GetScriptLoader()->ProcessPendingRequests();
    }
  }

  aRequest->GetLoadContext()->MaybeUnblockOnload();
}

nsresult ModuleLoader::CompileOrFinishModuleScript(
    JSContext* aCx, JS::Handle<JSObject*> aGlobal, JS::CompileOptions& aOptions,
    ModuleLoadRequest* aRequest, JS::MutableHandle<JSObject*> aModule) {
  if (aRequest->GetLoadContext()->mWasCompiledOMT) {
    JS::Rooted<JS::InstantiationStorage> storage(aCx);

    RefPtr<JS::Stencil> stencil;
    if (aRequest->IsTextSource()) {
      stencil = JS::FinishCompileModuleToStencilOffThread(
          aCx, aRequest->GetLoadContext()->mOffThreadToken, storage.address());
    } else {
      MOZ_ASSERT(aRequest->IsBytecode());
      stencil = JS::FinishDecodeStencilOffThread(
          aCx, aRequest->GetLoadContext()->mOffThreadToken, storage.address());
    }

    aRequest->GetLoadContext()->mOffThreadToken = nullptr;

    if (!stencil) {
      return NS_ERROR_FAILURE;
    }

    JS::InstantiateOptions instantiateOptions(aOptions);
    aModule.set(JS::InstantiateModuleStencil(aCx, instantiateOptions, stencil,
                                             storage.address()));
    if (!aModule) {
      return NS_ERROR_FAILURE;
    }

    if (aRequest->IsTextSource() &&
        ScriptLoader::ShouldCacheBytecode(aRequest)) {
      if (!JS::StartIncrementalEncoding(aCx, std::move(stencil))) {
        return NS_ERROR_FAILURE;
      }
    }

    return NS_OK;
  }

  if (!nsJSUtils::IsScriptable(aGlobal)) {
    return NS_OK;
  }

  RefPtr<JS::Stencil> stencil;
  if (aRequest->IsTextSource()) {
    MaybeSourceText maybeSource;
    nsresult rv = aRequest->GetScriptSource(aCx, &maybeSource);
    NS_ENSURE_SUCCESS(rv, rv);

    stencil = maybeSource.constructed<SourceText<char16_t>>()
                  ? JS::CompileModuleScriptToStencil(
                        aCx, aOptions, maybeSource.ref<SourceText<char16_t>>())
                  : JS::CompileModuleScriptToStencil(
                        aCx, aOptions, maybeSource.ref<SourceText<Utf8Unit>>());
  } else {
    MOZ_ASSERT(aRequest->IsBytecode());
    JS::DecodeOptions decodeOptions(aOptions);
    decodeOptions.borrowBuffer = true;

    auto& bytecode = aRequest->mScriptBytecode;
    auto& offset = aRequest->mBytecodeOffset;

    JS::TranscodeRange range(bytecode.begin() + offset,
                             bytecode.length() - offset);

    JS::TranscodeResult tr =
        JS::DecodeStencil(aCx, decodeOptions, range, getter_AddRefs(stencil));
    if (tr != JS::TranscodeResult::Ok) {
      return NS_ERROR_DOM_JS_DECODING_ERROR;
    }
  }

  if (!stencil) {
    return NS_ERROR_FAILURE;
  }

  JS::InstantiateOptions instantiateOptions(aOptions);
  aModule.set(JS::InstantiateModuleStencil(aCx, instantiateOptions, stencil));
  if (!aModule) {
    return NS_ERROR_FAILURE;
  }

  if (aRequest->IsTextSource() && ScriptLoader::ShouldCacheBytecode(aRequest)) {
    if (!JS::StartIncrementalEncoding(aCx, std::move(stencil))) {
      return NS_ERROR_FAILURE;
    }
  }

  return NS_OK;
}

/* static */
already_AddRefed<ModuleLoadRequest> ModuleLoader::CreateTopLevel(
    nsIURI* aURI, ScriptFetchOptions* aFetchOptions,
    const SRIMetadata& aIntegrity, nsIURI* aReferrer, ScriptLoader* aLoader,
    ScriptLoadContext* aContext) {
  RefPtr<ModuleLoadRequest> request = new ModuleLoadRequest(
      aURI, aFetchOptions, aIntegrity, aReferrer, aContext, true,
      /* is top level */ false, /* is dynamic import */
      aLoader->GetModuleLoader(),
      ModuleLoadRequest::NewVisitedSetForTopLevelImport(aURI), nullptr);

  return request.forget();
}

already_AddRefed<ModuleLoadRequest> ModuleLoader::CreateStaticImport(
    nsIURI* aURI, ModuleLoadRequest* aParent) {
  RefPtr<ScriptLoadContext> newContext =
      new ScriptLoadContext(aParent->GetLoadContext()->mElement);
  newContext->mIsInline = false;
  // Propagated Parent values. TODO: allow child modules to use root module's
  // script mode.
  newContext->mScriptMode = aParent->GetLoadContext()->mScriptMode;

  RefPtr<ModuleLoadRequest> request = new ModuleLoadRequest(
      aURI, aParent->mFetchOptions, SRIMetadata(), aParent->mURI, newContext,
      false, /* is top level */
      false, /* is dynamic import */
      aParent->mLoader, aParent->mVisitedSet, aParent->GetRootModule());

  return request.forget();
}

already_AddRefed<ModuleLoadRequest> ModuleLoader::CreateDynamicImport(
    JSContext* aCx, nsIURI* aURI, LoadedScript* aMaybeActiveScript,
    JS::Handle<JS::Value> aReferencingPrivate, JS::Handle<JSString*> aSpecifier,
    JS::Handle<JSObject*> aPromise) {
  MOZ_ASSERT(aSpecifier);
  MOZ_ASSERT(aPromise);

  RefPtr<ScriptFetchOptions> options;
  nsIURI* baseURL = nullptr;
  RefPtr<ScriptLoadContext> context;

  if (aMaybeActiveScript) {
    options = aMaybeActiveScript->GetFetchOptions();
    baseURL = aMaybeActiveScript->BaseURL();
    nsCOMPtr<Element> element = aMaybeActiveScript->GetScriptElement();
    context = new ScriptLoadContext(element);
  } else {
    // We don't have a referencing script so fall back on using
    // options from the document. This can happen when the user
    // triggers an inline event handler, as there is no active script
    // there.
    Document* document = GetScriptLoader()->GetDocument();

    // Use the document's principal for all loads, except WebExtension
    // content-scripts.
    // Only remember the global for content-scripts as well.
    nsCOMPtr<nsIPrincipal> principal = nsContentUtils::SubjectPrincipal(aCx);
    nsCOMPtr<nsIGlobalObject> global = xpc::CurrentNativeGlobal(aCx);
    if (!BasePrincipal::Cast(principal)->ContentScriptAddonPolicy()) {
      principal = document->NodePrincipal();
      MOZ_ASSERT(global);
      global = nullptr;  // Null global is the usual case for most loads.
    } else {
      MOZ_ASSERT(
          xpc::IsWebExtensionContentScriptSandbox(global->GetGlobalJSObject()));
    }

    options = new ScriptFetchOptions(
        mozilla::CORS_NONE, document->GetReferrerPolicy(), principal, global);
    baseURL = document->GetDocBaseURI();
    context = new ScriptLoadContext(nullptr);
  }

  context->mIsInline = false;
  context->mScriptMode = ScriptLoadContext::ScriptMode::eAsync;

  RefPtr<ModuleLoadRequest> request = new ModuleLoadRequest(
      aURI, options, SRIMetadata(), baseURL, context, true,
      /* is top level */ true, /* is dynamic import */
      this, ModuleLoadRequest::NewVisitedSetForTopLevelImport(aURI), nullptr);

  request->mDynamicReferencingPrivate = aReferencingPrivate;
  request->mDynamicSpecifier = aSpecifier;
  request->mDynamicPromise = aPromise;

  HoldJSObjects(request.get());

  return request.forget();
}

ModuleLoader::~ModuleLoader() {
  LOG(("ModuleLoader::~ModuleLoader %p", this));
  mLoader = nullptr;
}

#undef LOG
#undef LOG_ENABLED

}  // namespace mozilla::dom
