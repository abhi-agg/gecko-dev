/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * nsIAutoCompleteResult and nsILoginAutoCompleteSearch implementations for saved logins.
 */

"use strict";

const EXPORTED_SYMBOLS = ["LoginAutoComplete", "LoginAutoCompleteResult"];

const { XPCOMUtils } = ChromeUtils.import(
  "resource://gre/modules/XPCOMUtils.jsm"
);
const { Services } = ChromeUtils.import("resource://gre/modules/Services.jsm");

ChromeUtils.defineModuleGetter(
  this,
  "AutoCompleteChild",
  "resource://gre/actors/AutoCompleteChild.jsm"
);
ChromeUtils.defineModuleGetter(
  this,
  "BrowserUtils",
  "resource://gre/modules/BrowserUtils.jsm"
);
ChromeUtils.defineModuleGetter(
  this,
  "InsecurePasswordUtils",
  "resource://gre/modules/InsecurePasswordUtils.jsm"
);
ChromeUtils.defineModuleGetter(
  this,
  "LoginFormFactory",
  "resource://gre/modules/LoginFormFactory.jsm"
);
ChromeUtils.defineModuleGetter(
  this,
  "LoginHelper",
  "resource://gre/modules/LoginHelper.jsm"
);
ChromeUtils.defineModuleGetter(
  this,
  "LoginManagerChild",
  "resource://gre/modules/LoginManagerChild.jsm"
);
ChromeUtils.defineModuleGetter(
  this,
  "NewPasswordModel",
  "resource://gre/modules/NewPasswordModel.jsm"
);
XPCOMUtils.defineLazyServiceGetter(
  this,
  "formFillController",
  "@mozilla.org/satchel/form-fill-controller;1",
  Ci.nsIFormFillController
);
XPCOMUtils.defineLazyPreferenceGetter(
  this,
  "SHOULD_SHOW_ORIGIN",
  "signon.showAutoCompleteOrigins"
);
XPCOMUtils.defineLazyGetter(this, "log", () => {
  return LoginHelper.createLogger("LoginAutoComplete");
});
XPCOMUtils.defineLazyGetter(this, "passwordMgrBundle", () => {
  return Services.strings.createBundle(
    "chrome://passwordmgr/locale/passwordmgr.properties"
  );
});
XPCOMUtils.defineLazyGetter(this, "dateAndTimeFormatter", () => {
  return new Services.intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
  });
});

function loginSort(formHostPort, a, b) {
  let maybeHostPortA = LoginHelper.maybeGetHostPortForURL(a.origin);
  let maybeHostPortB = LoginHelper.maybeGetHostPortForURL(b.origin);
  if (formHostPort == maybeHostPortA && formHostPort != maybeHostPortB) {
    return -1;
  }
  if (formHostPort != maybeHostPortA && formHostPort == maybeHostPortB) {
    return 1;
  }

  if (a.httpRealm !== b.httpRealm) {
    // Sort HTTP auth. logins after form logins for the same origin.
    if (b.httpRealm === null) {
      return 1;
    }
    if (a.httpRealm === null) {
      return -1;
    }
  }

  let userA = a.username.toLowerCase();
  let userB = b.username.toLowerCase();

  if (userA < userB) {
    return -1;
  }

  if (userA > userB) {
    return 1;
  }

  return 0;
}

function findDuplicates(loginList) {
  let seen = new Set();
  let duplicates = new Set();
  for (let login of loginList) {
    if (seen.has(login.username)) {
      duplicates.add(login.username);
    }
    seen.add(login.username);
  }
  return duplicates;
}

function getLocalizedString(key, ...formatArgs) {
  if (formatArgs.length) {
    return passwordMgrBundle.formatStringFromName(key, formatArgs);
  }
  return passwordMgrBundle.GetStringFromName(key);
}

class AutocompleteItem {
  constructor(style) {
    this.comment = "";
    this.style = style;
    this.value = "";
  }

  removeFromStorage() {
    /* Do nothing by default */
  }
}

class InsecureLoginFormAutocompleteItem extends AutocompleteItem {
  constructor() {
    super("insecureWarning");

    XPCOMUtils.defineLazyGetter(this, "label", () => {
      let learnMoreString = getLocalizedString("insecureFieldWarningLearnMore");
      return getLocalizedString(
        "insecureFieldWarningDescription2",
        learnMoreString
      );
    });
  }
}

class LoginAutocompleteItem extends AutocompleteItem {
  login;
  #actor;

  constructor(
    login,
    hasBeenTypePassword,
    duplicateUsernames,
    actor,
    isOriginMatched
  ) {
    super(SHOULD_SHOW_ORIGIN ? "loginWithOrigin" : "login");
    this.login = login.QueryInterface(Ci.nsILoginMetaInfo);
    this.#actor = actor;

    let isDuplicateUsername =
      login.username && duplicateUsernames.has(login.username);

    XPCOMUtils.defineLazyGetter(this, "label", () => {
      let username = login.username;
      // If login is empty or duplicated we want to append a modification date to it.
      if (!username || isDuplicateUsername) {
        if (!username) {
          username = getLocalizedString("noUsername");
        }
        let time = dateAndTimeFormatter.format(
          new Date(login.timePasswordChanged)
        );
        username = getLocalizedString("loginHostAge", username, time);
      }
      return username;
    });

    XPCOMUtils.defineLazyGetter(this, "value", () => {
      return hasBeenTypePassword ? login.password : login.username;
    });

    XPCOMUtils.defineLazyGetter(this, "comment", () => {
      return JSON.stringify({
        guid: login.guid,
        login,
        isDuplicateUsername,
        isOriginMatched,
        comment:
          isOriginMatched && login.httpRealm === null
            ? getLocalizedString("displaySameOrigin")
            : login.displayOrigin,
      });
    });
  }

  removeFromStorage() {
    if (this.#actor) {
      let vanilla = LoginHelper.loginToVanillaObject(this.login);
      this.#actor.sendAsyncMessage("PasswordManager:removeLogin", {
        login: vanilla,
      });
    } else {
      Services.logins.removeLogin(this.login);
    }
  }
}

class GeneratedPasswordAutocompleteItem extends AutocompleteItem {
  constructor(generatedPassword, willAutoSaveGeneratedPassword) {
    super("generatedPassword");
    XPCOMUtils.defineLazyGetter(this, "comment", () => {
      return JSON.stringify({
        generatedPassword,
        willAutoSaveGeneratedPassword,
      });
    });
    this.value = generatedPassword;

    XPCOMUtils.defineLazyGetter(this, "label", () => {
      return getLocalizedString("useASecurelyGeneratedPassword");
    });
  }
}

class ImportableLearnMoreAutocompleteItem extends AutocompleteItem {
  constructor() {
    super("importableLearnMore");
  }
}

class ImportableLoginsAutocompleteItem extends AutocompleteItem {
  #actor;

  constructor(browserId, hostname, actor) {
    super("importableLogins");
    this.label = browserId;
    this.comment = hostname;
    this.#actor = actor;

    // This is sent for every item (re)shown, but the parent will debounce to
    // reduce the count by 1 total.
    this.#actor.sendAsyncMessage(
      "PasswordManager:decreaseSuggestImportCount",
      1
    );
  }

  removeFromStorage() {
    this.#actor.sendAsyncMessage(
      "PasswordManager:decreaseSuggestImportCount",
      100
    );
  }
}

class LoginsFooterAutocompleteItem extends AutocompleteItem {
  constructor(formHostname, telemetryEventData) {
    super("loginsFooter");
    XPCOMUtils.defineLazyGetter(this, "comment", () => {
      // The comment field of `loginsFooter` results have many additional pieces of
      // information for telemetry purposes. After bug 1555209, this information
      // can be passed to the parent process outside of nsIAutoCompleteResult APIs
      // so we won't need this hack.
      return JSON.stringify({
        ...telemetryEventData,
        formHostname,
      });
    });

    XPCOMUtils.defineLazyGetter(this, "label", () => {
      return getLocalizedString("viewSavedLogins.label");
    });
  }
}

// nsIAutoCompleteResult implementation
class LoginAutoCompleteResult {
  #rows = [];

  constructor(
    aSearchString,
    matchingLogins,
    formOrigin,
    {
      generatedPassword,
      willAutoSaveGeneratedPassword,
      importable,
      isSecure,
      actor,
      hasBeenTypePassword,
      hostname,
      telemetryEventData,
    }
  ) {
    let hidingFooterOnPWFieldAutoOpened = false;
    const importableBrowsers =
      importable?.state === "import" && importable?.browsers;

    function isFooterEnabled() {
      // We need to check LoginHelper.enabled here since the insecure warning should
      // appear even if pwmgr is disabled but the footer should never appear in that case.
      if (!LoginHelper.showAutoCompleteFooter || !LoginHelper.enabled) {
        return false;
      }

      // Don't show the footer on non-empty password fields as it's not providing
      // value and only adding noise since a password was already filled.
      if (hasBeenTypePassword && aSearchString && !generatedPassword) {
        log.debug("Hiding footer: non-empty password field");
        return false;
      }

      if (
        !importableBrowsers &&
        !matchingLogins.length &&
        !generatedPassword &&
        hasBeenTypePassword &&
        formFillController.passwordPopupAutomaticallyOpened
      ) {
        hidingFooterOnPWFieldAutoOpened = true;
        log.debug(
          "Hiding footer: no logins and the popup was opened upon focus of the pw. field"
        );
        return false;
      }

      return true;
    }

    this.searchString = aSearchString;

    // Insecure field warning comes first if it applies and is enabled.
    if (!isSecure && LoginHelper.showInsecureFieldWarning) {
      this.#rows.push(new InsecureLoginFormAutocompleteItem());
    }

    // Saved login items
    let formHostPort = LoginHelper.maybeGetHostPortForURL(formOrigin);
    let logins = matchingLogins.sort(loginSort.bind(null, formHostPort));
    let duplicateUsernames = findDuplicates(matchingLogins);

    for (let login of logins) {
      let item = new LoginAutocompleteItem(
        login,
        hasBeenTypePassword,
        duplicateUsernames,
        actor,
        LoginHelper.isOriginMatching(login.origin, formOrigin, {
          schemeUpgrades: LoginHelper.schemeUpgrades,
        })
      );
      this.#rows.push(item);
    }

    // The footer comes last if it's enabled
    if (isFooterEnabled()) {
      if (generatedPassword) {
        this.#rows.push(
          new GeneratedPasswordAutocompleteItem(
            generatedPassword,
            willAutoSaveGeneratedPassword
          )
        );
      }

      // Suggest importing logins if there are none found.
      if (!logins.length && importableBrowsers) {
        this.#rows.push(
          ...importableBrowsers.map(
            browserId =>
              new ImportableLoginsAutocompleteItem(browserId, hostname, actor)
          )
        );
        this.#rows.push(new ImportableLearnMoreAutocompleteItem());
      }

      this.#rows.push(
        new LoginsFooterAutocompleteItem(hostname, telemetryEventData)
      );
    }

    // Determine the result code and default index.
    if (this.matchCount > 0) {
      this.searchResult = Ci.nsIAutoCompleteResult.RESULT_SUCCESS;
      this.defaultIndex = 0;
    } else if (hidingFooterOnPWFieldAutoOpened) {
      // We use a failure result so that the empty results aren't re-used for when
      // the user tries to manually open the popup (we want the footer in that case).
      this.searchResult = Ci.nsIAutoCompleteResult.RESULT_FAILURE;
      this.defaultIndex = -1;
    }
  }

  QueryInterface = ChromeUtils.generateQI([
    "nsIAutoCompleteResult",
    "nsISupportsWeakReference",
  ]);

  /**
   * Accessed via .wrappedJSObject
   * @private
   */
  get logins() {
    return this.#rows
      .filter(item => item instanceof LoginAutocompleteItem)
      .map(item => item.login);
  }

  // Allow autoCompleteSearch to get at the JS object so it can
  // modify some readonly properties for internal use.
  get wrappedJSObject() {
    return this;
  }

  // Interfaces from idl...
  searchString = null;
  searchResult = Ci.nsIAutoCompleteResult.RESULT_NOMATCH;
  defaultIndex = -1;
  errorDescription = "";

  get matchCount() {
    return this.#rows.length;
  }

  #throwOnBadIndex(index) {
    if (index < 0 || index >= this.matchCount) {
      throw new Error("Index out of range.");
    }
  }

  getValueAt(index) {
    this.#throwOnBadIndex(index);
    return this.#rows[index].value;
  }

  getLabelAt(index) {
    this.#throwOnBadIndex(index);
    return this.#rows[index].label;
  }

  getCommentAt(index) {
    this.#throwOnBadIndex(index);
    return this.#rows[index].comment;
  }

  getStyleAt(index) {
    this.#throwOnBadIndex(index);
    return this.#rows[index].style;
  }

  getImageAt(index) {
    this.#throwOnBadIndex(index);
    return "";
  }

  getFinalCompleteValueAt(index) {
    return this.getValueAt(index);
  }

  isRemovableAt(index) {
    this.#throwOnBadIndex(index);
    return true;
  }

  removeValueAt(index) {
    this.#throwOnBadIndex(index);

    let [removedItem] = this.#rows.splice(index, 1);

    if (this.defaultIndex > this.#rows.length) {
      this.defaultIndex--;
    }

    removedItem.removeFromStorage();
  }
}

class LoginAutoComplete {
  // HTMLInputElement to number, the element's new-password heuristic confidence score
  #cachedNewPasswordScore = new WeakMap();
  #autoCompleteLookupPromise = null;
  classID = Components.ID("{2bdac17c-53f1-4896-a521-682ccdeef3a8}");
  QueryInterface = ChromeUtils.generateQI(["nsILoginAutoCompleteSearch"]);

  /**
   * Yuck. This is called directly by satchel:
   * nsFormFillController::StartSearch()
   * [toolkit/components/satchel/nsFormFillController.cpp]
   *
   * We really ought to have a simple way for code to register an
   * auto-complete provider, and not have satchel calling pwmgr directly.
   *
   * @param {string} aSearchString The value typed in the field.
   * @param {nsIAutoCompleteResult} aPreviousResult
   * @param {HTMLInputElement} aElement
   * @param {nsIFormAutoCompleteObserver} aCallback
   */
  startSearch(aSearchString, aPreviousResult, aElement, aCallback) {
    let { isNullPrincipal } = aElement.nodePrincipal;
    if (aElement.nodePrincipal.schemeIs("about")) {
      // Don't show autocomplete results for about: pages.
      // XXX: Don't we need to call the callback here?
      return;
    }

    let searchStartTimeMS = Services.telemetry.msSystemNow();

    // Show the insecure login warning in the passwords field on null principal documents.
    // Avoid loading InsecurePasswordUtils.jsm in a sandboxed document (e.g. an ad. frame) if we
    // already know it has a null principal and will therefore get the insecure autocomplete
    // treatment.
    // InsecurePasswordUtils doesn't handle the null principal case as not secure because we don't
    // want the same treatment:
    // * The web console warnings will be confusing (as they're primarily about http:) and not very
    //   useful if the developer intentionally sandboxed the document.
    // * The site identity insecure field warning would require LoginManagerChild being loaded and
    //   listening to some of the DOM events we're ignoring in null principal documents. For memory
    //   reasons it's better to not load LMC at all for these sandboxed frames. Also, if the top-
    //   document is sandboxing a document, it probably doesn't want that sandboxed document to be
    //   able to affect the identity icon in the address bar by adding a password field.
    let form = LoginFormFactory.createFromField(aElement);
    let isSecure = !isNullPrincipal && InsecurePasswordUtils.isFormSecure(form);
    let { hasBeenTypePassword } = aElement;
    let hostname = aElement.ownerDocument.documentURIObject.host;
    let formOrigin = LoginHelper.getLoginOrigin(
      aElement.ownerDocument.documentURI
    );
    let loginManagerActor = LoginManagerChild.forWindow(aElement.ownerGlobal);
    let completeSearch = async autoCompleteLookupPromise => {
      // Assign to the member synchronously before awaiting the Promise.
      this.#autoCompleteLookupPromise = autoCompleteLookupPromise;

      let {
        generatedPassword,
        importable,
        logins,
        willAutoSaveGeneratedPassword,
      } = await autoCompleteLookupPromise;

      // If the search was canceled before we got our
      // results, don't bother reporting them.
      // N.B. This check must occur after the `await` above for it to be
      // effective.
      if (this.#autoCompleteLookupPromise !== autoCompleteLookupPromise) {
        log.debug("ignoring result from previous search");
        return;
      }

      let telemetryEventData = {
        acFieldName: aElement.getAutocompleteInfo().fieldName,
        hadPrevious: !!aPreviousResult,
        typeWasPassword: aElement.hasBeenTypePassword,
        fieldType: aElement.type,
        searchStartTimeMS,
        stringLength: aSearchString.length,
      };

      this.#autoCompleteLookupPromise = null;
      let results = new LoginAutoCompleteResult(
        aSearchString,
        logins,
        formOrigin,
        {
          generatedPassword,
          willAutoSaveGeneratedPassword,
          importable,
          actor: loginManagerActor,
          isSecure,
          hasBeenTypePassword,
          hostname,
          telemetryEventData,
        }
      );
      aCallback.onSearchCompletion(results);
    };

    if (isNullPrincipal) {
      // Don't search login storage when the field has a null principal as we don't want to fill
      // logins for the `location` in this case.
      completeSearch(Promise.resolve({ logins: [] }));
      return;
    }

    if (
      hasBeenTypePassword &&
      aSearchString &&
      !loginManagerActor.isPasswordGenerationForcedOn(aElement)
    ) {
      // Return empty result on password fields with password already filled,
      // unless password generation was forced.
      completeSearch(Promise.resolve({ logins: [] }));
      return;
    }

    if (!LoginHelper.enabled) {
      completeSearch(Promise.resolve({ logins: [] }));
      return;
    }

    let previousResult;
    if (aPreviousResult) {
      previousResult = {
        searchString: aPreviousResult.searchString,
        logins: LoginHelper.loginsToVanillaObjects(
          aPreviousResult.wrappedJSObject.logins
        ),
      };
    } else {
      previousResult = null;
    }

    let acLookupPromise = this.#requestAutoCompleteResultsFromParent({
      searchString: aSearchString,
      previousResult,
      inputElement: aElement,
      form,
      hasBeenTypePassword,
    });
    completeSearch(acLookupPromise).catch(log.error.bind(log));
  }

  stopSearch() {
    this.#autoCompleteLookupPromise = null;
  }

  async #requestAutoCompleteResultsFromParent({
    searchString,
    previousResult,
    inputElement,
    form,
    hasBeenTypePassword,
  }) {
    let actionOrigin = LoginHelper.getFormActionOrigin(form);
    let autocompleteInfo = inputElement.getAutocompleteInfo();

    let loginManagerActor = LoginManagerChild.forWindow(
      inputElement.ownerGlobal
    );
    let forcePasswordGeneration = false;
    let isProbablyANewPasswordField = false;
    if (hasBeenTypePassword) {
      forcePasswordGeneration = loginManagerActor.isPasswordGenerationForcedOn(
        inputElement
      );
      // Run the Fathom model only if the password field does not have the
      // autocomplete="new-password" attribute.
      isProbablyANewPasswordField =
        autocompleteInfo.fieldName == "new-password" ||
        this.isProbablyANewPasswordField(inputElement);
    }

    let messageData = {
      actionOrigin,
      searchString,
      previousResult,
      forcePasswordGeneration,
      hasBeenTypePassword,
      isSecure: InsecurePasswordUtils.isFormSecure(form),
      isProbablyANewPasswordField,
    };

    if (LoginHelper.showAutoCompleteFooter) {
      gAutoCompleteListener.init();
    }

    log.debug("LoginAutoComplete search:", {
      forcePasswordGeneration,
      isSecure: messageData.isSecure,
      hasBeenTypePassword,
      isProbablyANewPasswordField,
      searchStringLength: searchString.length,
    });

    let result = await loginManagerActor.sendQuery(
      "PasswordManager:autoCompleteLogins",
      messageData
    );

    return {
      generatedPassword: result.generatedPassword,
      importable: result.importable,
      logins: LoginHelper.vanillaObjectsToLogins(result.logins),
      willAutoSaveGeneratedPassword: result.willAutoSaveGeneratedPassword,
    };
  }

  isProbablyANewPasswordField(inputElement) {
    const threshold = LoginHelper.generationConfidenceThreshold;
    if (threshold == -1) {
      // Fathom is disabled
      return false;
    }

    let score = this.#cachedNewPasswordScore.get(inputElement);
    if (score) {
      return score >= threshold;
    }

    const { rules, type } = NewPasswordModel;
    const results = rules.against(inputElement);
    score = results.get(inputElement).scoreFor(type);
    this.#cachedNewPasswordScore.set(inputElement, score);
    return score >= threshold;
  }
}

let gAutoCompleteListener = {
  // Input element on which enter keydown event was fired.
  keyDownEnterForInput: null,

  added: false,

  init() {
    if (!this.added) {
      AutoCompleteChild.addPopupStateListener(this);
      this.added = true;
    }
  },

  popupStateChanged(messageName, data, target) {
    switch (messageName) {
      case "FormAutoComplete:PopupOpened": {
        let { chromeEventHandler } = target.docShell;
        chromeEventHandler.addEventListener("keydown", this, true);
        break;
      }

      case "FormAutoComplete:PopupClosed": {
        this.onPopupClosed(data, target);
        let { chromeEventHandler } = target.docShell;
        chromeEventHandler.removeEventListener("keydown", this, true);
        break;
      }
    }
  },

  handleEvent(event) {
    if (event.type != "keydown") {
      return;
    }

    let focusedElement = formFillController.focusedInput;
    if (
      event.keyCode != event.DOM_VK_RETURN ||
      focusedElement != event.target
    ) {
      this.keyDownEnterForInput = null;
      return;
    }
    this.keyDownEnterForInput = focusedElement;
  },

  onPopupClosed({ selectedRowComment, selectedRowStyle }, window) {
    let focusedElement = formFillController.focusedInput;
    let eventTarget = this.keyDownEnterForInput;
    this.keyDownEnterForInput = null;
    if (!eventTarget || eventTarget !== focusedElement) {
      return;
    }

    let loginManager = window.windowGlobalChild.getActor("LoginManager");
    switch (selectedRowStyle) {
      case "importableLearnMore":
        loginManager.sendAsyncMessage(
          "PasswordManager:OpenImportableLearnMore",
          {}
        );
        break;
      case "importableLogins":
        loginManager.sendAsyncMessage("PasswordManager:HandleImportable", {
          browserId: selectedRowComment,
        });
        break;
      case "loginsFooter":
        loginManager.sendAsyncMessage("PasswordManager:OpenPreferences", {
          entryPoint: "autocomplete",
        });
        break;
    }
  },
};
