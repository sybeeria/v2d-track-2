# Frozen label-free reconstruction review protocol

Written before new ep18 measurements on 2026-10-22. These are conservative
engineering triage thresholds [G], not validated predictors of policy success.
No threshold will be fitted to ep18 or a challenge clean trajectory.

Inputs: video, generated Tier 3 references/initializers, video-derived masks,
camera calibration, and released scene geometry only under the existing
engineering-only/ruling-pending condition. No captions, Tier 2 motion, or later
Tier 1 frames. This review needs no new Tier 1 read; existing alignment is reused.

1. Complete finite rigid poses and matching frame counts are mandatory.
2. Full mesh support distance: flag penetration below -0.02 m or an object
   bottom above +0.03 m without geometric hand contact for >=6 source frames.
   Use inferred human fingertips <=0.03 m from the mesh as a contact candidate;
   >=0.08 m while the object moves (>0.03 m/s, 15-frame smoothed translation)
   for >=6 frames is a hand-association flag. Report raw and smoothed velocities.
   A hand gap is neither a force nor a grasp. Feet/other supports require review.
3. At native 30 Hz and actual 50 Hz playback, flag object translation speed
   >2 m/s, acceleration >20 m/s^2, angular speed >10 rad/s or angular
   acceleration >100 rad/s^2. Report p95/max and exact intervals. A duration of
   >=3 frames makes a triage failure; a single spike still requires inspection.
4. Compare projected mesh silhouette to a video-only mask at evenly spaced
   frames plus kinematic extrema. Report IoU and centroid error normalized by
   mask box diagonal. Fail independent visible-object checks if IoU <0.5 or
   normalized centroid error >0.1 on >10% of checked visible frames. Occlusion
   is recorded separately; propagated SAM2 is not independent ground truth.
5. Repair candidates fixed here: (a) support projection only in video-supported
   ground phases, (b) existing 15-frame cubic translation/rotation smoothing
   with the same shift applied to the human root to preserve relative geometry,
   (c) global retiming with factor >=max(vmax/2,sqrt(amax/20)) at 50 Hz.
   Retiming is diagnostic only: an unchanged full-video correspondence is
   required for production, so slowing index playback is not an accepted fix.
   Never snap all airborne frames solely because a reconstructed hand is far.
6. A candidate may proceed to a GPU pilot only if coverage/provenance pass,
   no sustained support/contact failures remain, no >=3-frame dynamics flags,
   and independent visible-object checks pass. Unknown image/contact evidence
   means HOLD, not PASS. Smoothing alone cannot certify reconstruction accuracy.
7. Keep ep18 full length, endpoint/time correspondences, all failed candidates,
   before/after image and physical metrics, source hashes and repair magnitude.
   Do not select by a self-score. Only an eventual final unassisted T2C-contact
   policy self-MPPE <=10 cm can pass the existing learnability veto; a passing
   self-score still says nothing about clean reconstruction accuracy.

All prototype motion outputs are diagnostic training references, never submitted
object rollouts. No physics change, new GPU job, packer or external write.
