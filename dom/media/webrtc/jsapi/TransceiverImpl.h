/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef _TRANSCEIVERIMPL_H_
#define _TRANSCEIVERIMPL_H_

#include <string>
#include "libwebrtcglue/MediaConduitControl.h"
#include "mozilla/RefPtr.h"
#include "nsCOMPtr.h"
#include "nsISerialEventTarget.h"
#include "nsTArray.h"
#include "mozilla/dom/MediaStreamTrack.h"
#include "ErrorList.h"
#include "jsep/JsepTransceiver.h"
#include "transport/transportlayer.h"  // For TransportLayer::State

class nsIPrincipal;

namespace mozilla {
class PeerIdentity;
class JsepTransceiver;
class MediaSessionConduit;
class VideoSessionConduit;
class AudioSessionConduit;
struct AudioCodecConfig;
class VideoCodecConfig;  // Why is this a class, but AudioCodecConfig a struct?
class MediaPipelineTransmit;
class MediaPipeline;
class MediaPipelineFilter;
class MediaTransportHandler;
class RTCStatsIdGenerator;
class WebrtcCallWrapper;
class JsepTrackNegotiatedDetails;

namespace dom {
class RTCDtlsTransport;
class RTCDTMFSender;
class RTCRtpTransceiver;
struct RTCRtpSourceEntry;
class RTCRtpReceiver;
}  // namespace dom

/**
 * This is what ties all the various pieces that make up a transceiver
 * together. This includes:
 * MediaStreamTrack for rendering and capture
 * MediaTransportHandler for RTP transmission/reception
 * Audio/VideoConduit for feeding RTP/RTCP into webrtc.org for decoding, and
 * feeding audio/video frames into webrtc.org for encoding into RTP/RTCP.
 */
class TransceiverImpl : public nsISupports,
                        public nsWrapperCache,
                        public sigslot::has_slots<> {
 public:
  /**
   * |aSendTrack| might or might not be set.
   */
  TransceiverImpl(
      nsPIDOMWindowInner* aWindow, bool aPrivacyNeeded,
      const std::string& aPCHandle, MediaTransportHandler* aTransportHandler,
      JsepTransceiver* aJsepTransceiver, nsISerialEventTarget* aMainThread,
      nsISerialEventTarget* aStsThread, dom::MediaStreamTrack* aSendTrack,
      WebrtcCallWrapper* aCallWrapper, RTCStatsIdGenerator* aIdGenerator);

  bool IsValid() const { return !!mConduit; }

  nsresult UpdateSendTrack(dom::MediaStreamTrack* aSendTrack);

  nsresult UpdateSinkIdentity(const dom::MediaStreamTrack* aTrack,
                              nsIPrincipal* aPrincipal,
                              const PeerIdentity* aSinkIdentity);

  nsresult UpdateTransport();

  nsresult UpdateConduit();

  void ResetSync();

  nsresult SyncWithMatchingVideoConduits(
      nsTArray<RefPtr<TransceiverImpl>>& transceivers);

  void Shutdown_m();

  bool ConduitHasPluginID(uint64_t aPluginID);

  bool HasSendTrack(const dom::MediaStreamTrack* aSendTrack) const;

  // This is so PCImpl can unregister from PrincipalChanged callbacks; maybe we
  // should have TransceiverImpl handle these callbacks instead? It would need
  // to be able to get a ref to PCImpl though.
  RefPtr<dom::MediaStreamTrack> GetSendTrack() { return mSendTrack; }

  // for webidl
  JSObject* WrapObject(JSContext* aCx,
                       JS::Handle<JSObject*> aGivenProto) override;
  nsPIDOMWindowInner* GetParentObject() const;
  void SyncWithJS(dom::RTCRtpTransceiver& aJsTransceiver, ErrorResult& aRv);
  dom::RTCRtpReceiver* Receiver() const { return mReceiver; }
  dom::RTCDTMFSender* GetDtmf() const { return mDtmf; }
  dom::RTCDtlsTransport* GetDtlsTransport() const { return mDtlsTransport; }

  bool CanSendDTMF() const;

  // TODO: These are for stats; try to find a cleaner way.
  RefPtr<MediaPipelineTransmit> GetSendPipeline();

  void UpdateDtlsTransportState(const std::string& aTransportId,
                                TransportLayer::State aState);
  void SetDtlsTransport(dom::RTCDtlsTransport* aDtlsTransport, bool aStable);
  void RollbackToStableDtlsTransport();

  std::string GetTransportId() const {
    return mJsepTransceiver->mTransport.mTransportId;
  }

  bool IsVideo() const;

  bool IsSending() const {
    return !mJsepTransceiver->IsStopped() &&
           mJsepTransceiver->mSendTrack.GetActive();
  }

  bool IsReceiving() const {
    return !mJsepTransceiver->IsStopped() &&
           mJsepTransceiver->mRecvTrack.GetActive();
  }

  Maybe<const std::vector<UniquePtr<JsepCodecDescription>>&>
  GetNegotiatedSendCodecs() const;

  Maybe<const std::vector<UniquePtr<JsepCodecDescription>>&>
  GetNegotiatedRecvCodecs() const;

  struct PayloadTypes {
    Maybe<int> mSendPayloadType;
    Maybe<int> mRecvPayloadType;
  };
  using ActivePayloadTypesPromise = MozPromise<PayloadTypes, nsresult, true>;
  RefPtr<ActivePayloadTypesPromise> GetActivePayloadTypes() const;

  MediaSessionConduit* GetConduit() const { return mConduit; }

  // nsISupports
  NS_DECL_CYCLE_COLLECTING_ISUPPORTS
  NS_DECL_CYCLE_COLLECTION_SCRIPT_HOLDER_CLASS(TransceiverImpl)

  static nsresult NegotiatedDetailsToAudioCodecConfigs(
      const JsepTrackNegotiatedDetails& aDetails,
      std::vector<AudioCodecConfig>* aConfigs);

  static nsresult NegotiatedDetailsToVideoCodecConfigs(
      const JsepTrackNegotiatedDetails& aDetails,
      std::vector<VideoCodecConfig>* aConfigs);

  /**
   * Takes a set of codec stats (per-peerconnection) and a set of
   * transceiver/transceiver-stats-promise tuples. Filters out all referenced
   * codec stats based on the transceiver's transport and rtp stream stats.
   * Finally returns the flattened stats containing the filtered codec stats and
   * all given per-transceiver-stats.
   */
  static RefPtr<dom::RTCStatsPromise> ApplyCodecStats(
      nsTArray<dom::RTCCodecStats> aCodecStats,
      nsTArray<std::tuple<TransceiverImpl*,
                          RefPtr<dom::RTCStatsPromise::AllPromiseType>>>
          aTransceiverStatsPromises);

  AbstractCanonical<bool>* CanonicalReceiving() { return &mReceiving; }
  AbstractCanonical<bool>* CanonicalTransmitting() { return &mTransmitting; }
  AbstractCanonical<Ssrcs>* CanonicalLocalSsrcs() { return &mLocalSsrcs; }
  AbstractCanonical<std::string>* CanonicalLocalCname() { return &mLocalCname; }
  AbstractCanonical<std::string>* CanonicalLocalMid() { return &mLocalMid; }
  AbstractCanonical<std::string>* CanonicalSyncGroup() { return &mSyncGroup; }
  AbstractCanonical<RtpExtList>* CanonicalLocalSendRtpExtensions() {
    return &mLocalSendRtpExtensions;
  }
  AbstractCanonical<Maybe<AudioCodecConfig>>* CanonicalAudioSendCodec() {
    return &mAudioSendCodec;
  }
  AbstractCanonical<Ssrcs>* CanonicalLocalVideoRtxSsrcs() {
    return &mLocalVideoRtxSsrcs;
  }
  AbstractCanonical<Maybe<VideoCodecConfig>>* CanonicalVideoSendCodec() {
    return &mVideoSendCodec;
  }
  AbstractCanonical<Maybe<RtpRtcpConfig>>* CanonicalVideoSendRtpRtcpConfig() {
    return &mVideoSendRtpRtcpConfig;
  }
  AbstractCanonical<webrtc::VideoCodecMode>* CanonicalVideoCodecMode() {
    return &mVideoCodecMode;
  }

 private:
  virtual ~TransceiverImpl();
  void InitAudio();
  void InitVideo();
  void InitConduitControl();
  nsresult UpdateAudioConduit();
  nsresult UpdateVideoConduit();
  nsresult ConfigureVideoCodecMode();
  void Stop();

  nsCOMPtr<nsPIDOMWindowInner> mWindow;
  const std::string mPCHandle;
  RefPtr<MediaTransportHandler> mTransportHandler;
  const RefPtr<JsepTransceiver> mJsepTransceiver;
  bool mHaveSetupTransport;
  nsCOMPtr<nsISerialEventTarget> mMainThread;
  nsCOMPtr<nsISerialEventTarget> mStsThread;
  RefPtr<dom::MediaStreamTrack> mSendTrack;
  // state for webrtc.org that is shared between all transceivers
  RefPtr<WebrtcCallWrapper> mCallWrapper;
  RefPtr<MediaSessionConduit> mConduit;
  // Call thread only.
  RefPtr<MediaPipelineTransmit> mTransmitPipeline;
  // The spec says both RTCRtpReceiver and RTCRtpSender have a slot for
  // an RTCDtlsTransport.  They are always the same, so we'll store it
  // here.
  RefPtr<dom::RTCDtlsTransport> mDtlsTransport;
  // The spec says both RTCRtpReceiver and RTCRtpSender have a slot for
  // a last stable state RTCDtlsTransport.  They are always the same, so
  // we'll store it here.
  RefPtr<dom::RTCDtlsTransport> mLastStableDtlsTransport;
  RefPtr<dom::RTCRtpReceiver> mReceiver;
  // TODO(bug 1616937): Move this to RTCRtpSender
  RefPtr<dom::RTCDTMFSender> mDtmf;

  Canonical<bool> mReceiving;
  Canonical<bool> mTransmitting;
  Canonical<Ssrcs> mLocalSsrcs;
  Canonical<Ssrcs> mLocalVideoRtxSsrcs;
  Canonical<std::string> mLocalCname;
  Canonical<std::string> mLocalMid;
  Canonical<std::string> mSyncGroup;
  Canonical<RtpExtList> mLocalSendRtpExtensions;

  Canonical<Maybe<AudioCodecConfig>> mAudioSendCodec;

  Canonical<Maybe<VideoCodecConfig>> mVideoSendCodec;
  Canonical<Maybe<RtpRtcpConfig>> mVideoSendRtpRtcpConfig;
  Canonical<webrtc::VideoCodecMode> mVideoCodecMode;
};

}  // namespace mozilla

#endif  // _TRANSCEIVERIMPL_H_
