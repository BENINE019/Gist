# Grasp

**Show it once. It grasps what you meant.**

Film yourself doing a task on your phone, once, while saying what you are doing.
Grasp works out what you meant, not just what you did, and then finishes the same
task in scenes it has never seen: objects moved, extra objects added, distractors
on the table. Every successful run becomes a robot training episode.

Built solo for the Meta Global AI Developer Hackathon (Oct 27 to Nov 4, 2026),
indie developer track, problem statement *Push the Frontier*.

- Demo video: *(link)*
- Live demo (upload your own clip, shuffle the scene): *(link)*
- Results: [eval/results.md](eval/results.md)

## The claim, in one picture

A robot that copies the demonstration fails the moment anything moves. A robot
that inferred the goal does not.

| | Original scene | 100 shuffled scenes |
| --- | --- | --- |
| Goal plan (Grasp) | 10 / 10 | **100%** |
| Copy the demonstration | 9 / 10 | **1%** |

Ten phone demonstrations in, 110 verified episodes out.

## How it works

Eight stages, each a module in `mimic/` with a JSON output, so every stage can be
inspected on its own.

1. **Phone clip.** 10 to 30 seconds, one printed ArUco marker on the table, the
   person narrating as they work.
2. **Name objects** (Muse Spark). A few sampled frames in, short noun phrases out:
   "red pencil", "white cup". No class list, no training.
3. **Track** (SAM 3.1 video). One request per phrase; one identity per object through
   hand occlusion. The hand is tracked too.
4. **Map to the table.** The marker gives a homography from pixels to centimetres on
   the table plane.
5. **Find events.** Rest periods between hand contacts become moves: grasp frame,
   release frame, from where, to where, into which container, onto what. Objects
   that move together are grouped (a keychain rides with its wallet). Objects that
   never move are context.
6. **Infer the goal** (Muse Spark, Muse Voice Transcribe). The first and last frame,
   the event list and the narration go in; a structured goal comes out, in a
   five-type vocabulary, with ranked candidates and confidences. If the top two are
   close, Grasp asks a clarifying question instead of guessing.
7. **Replay and plan** (MuJoCo, Franka Panda). Two robots in the demonstrated scene:
   one copies the recorded motions, one plans from the goal.
8. **Generalize.** The scene is shuffled; every second scene gets an extra object of
   the demonstrated kind, every third a distractor. Both robots run, success is
   checked, and successful goal runs are exported as LeRobot v2.1 episodes.

### Goal vocabulary

```
put_all(category, container)     all pencils in the cup
move(object, relation, anchor)   the wallet next to the cup; the eraser to the leftmost position
arrange(category, pattern)       the four objects in a row
stack(top, base)                 the small eraser on the big one
swap(a, b)                       the scissors and the glasses change places
```

Each goal also carries `scope` (just this object, or every object of its kind) and
`order_matters`. Those two fields are where most of the ambiguity lives, and where the
narration decides.

## Meta models used

| Model | Role |
| --- | --- |
| **SAM 3.1** (Model API, video) | Tracks every named object and the hand through the clip. Identity through occlusion is what the whole idea rests on. |
| **Muse Spark** (Model API) | Names objects from frames; infers the goal from events and narration; writes the clarifying question. |
| **Muse Voice Transcribe** (Model API) | Turns the narration into timestamped text, the main signal for resolving goal ambiguity. |

## Results

Ten clips recorded during the hackathon, ground truth written before processing
(`clips/goal.md`). Full tables with per-clip detail in [eval/results.md](eval/results.md).

**Perception.** Naming recall 41 of 42 objects (98%). The miss was a keychain. SAM found
four objects under two names each; a track-overlap merge fixed all four.

**Events.** 22 of 23 moved objects detected (96%); the miss was an object SAM never
named. Keychain correctly grouped with the wallet in both clips where it was attached.

**Goal inference.** With narration: goal type correct on 10 of 10 clips, scope on 9 of
10, no clarifying question needed. Without narration: type still 10 of 10, but the model
offered two readings and had to ask on 4 of 10 clips. Narration mostly buys certainty.

**Generalization.** Across 100 shuffled scenes (10 per clip): goal plan 100%, copy
baseline 1%. The one copy success was a layout where the object happened to land within
3 cm of its demonstrated pick position. Scenes where the goal already held before acting
were rerolled. In the original scene both robots succeed on 9 of 10 clips; the copy robot
misses once on a 4 cm tolerance versus the human's own placement.

**Dataset.** 110 verified episodes, 45,792 frames, 10 task sentences, exported in the
LeRobot v2.1 layout (`python -m mimic.export --name grasp_v1`). Each frame holds seven
joint angles plus gripper opening as `observation.state`, the next frame as `action`,
and a 640 x 480 video from a fixed camera.

## Quick start

```bash
git clone https://github.com/BENINE019/Grasp.git && cd Grasp
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
export MODEL_API_KEY="..."          # Meta Model API key; never commit it

# one clip, end to end
bash scripts/run_all.sh 9.5          # marker side length in cm; naming, tracking, table, events, narration, goal
MUJOCO_GL=egl python -m mimic.generalize clips/c01.mp4 --scenes 10 --render 2
python -m mimic.eval

# the web demo
MUJOCO_GL=egl uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Recording your own clip: phone propped 60 cm above the table at about 45 degrees, the
printed marker (DICT_4X4_50, id 0) flat in a corner and never covered, every object
visible in the first second, narrate as you go. Convert with `scripts/convert_clips.sh`.

## Repository

```
mimic/            the pipeline (package name kept from the project's first working name)
  tracking.py     stage 3   SAM 3.1 tracks
  table_map.py    stage 4   marker calibration
  events.py       stage 5   moves, containers, groups
  transcribe.py   stage 6a  narration
  intent.py       stage 6b  goal inference
  planner.py      goal to steps, and the success check
  sim.py          MuJoCo scene, IK, kinematic replay
  generalize.py   shuffled scenes, goal vs copy
  export.py       LeRobot v2.1 dataset
  eval.py         the four tables
scripts/          naming (name_objects.py), clip conversion, batch runner, the Oct 27 sanity checks
app/              FastAPI server and the single-page demo
clips/            the ten demonstrations, narration audio, ground truth (goal.md)
eval/             results.md, results.json
prompts live in intent.py and scripts/name_objects.py
```

## Related work

Learning manipulation from human video is an active research area. ORION builds
open-world object graphs from a single human video; OKAMI retargets a single video to a
humanoid; Track2Act, ATM and Im2Flow2Act turn predicted object or point tracks into
actions; MimicPlay, Phantom and EgoMimic train policies from human video, sometimes with
dedicated capture rigs; UMI and DexCap collect demonstrations without a robot but with
custom hardware.

Grasp is not a new method. Its difference is setup and scope: a phone and one printed
marker, no task-specific training, narration as the disambiguating signal, a goal-level
representation that generalizes across layouts, and an end-to-end tool built on hosted
foundation models in eight days.

## Limitations

Stated plainly, because they shape what the results mean.

- **Kinematic replay with a scripted grasp.** The simulated gripper does not pinch
  objects with contact physics; an object follows the gripper once reached. Long thin
  objects are tilted upright before going into a container.
- **Table plane only.** One phone camera gives no depth, so positions are 2D on the table
  through the marker. Object orientation is not recovered; footprints come from a
  per-category table.
- **Success is judged against simulator state**, not by a vision check of the render.
- **Five goal types.** Tasks outside them (pouring, folding, pressing) are out of scope.
- **Naming misses.** Muse Spark and SAM 3.1 missed a pencil lead case and, in two clips,
  a keychain. A misnamed sheet of paper is filtered because it covers the marker.
- **Narration is doing real work.** Without it, 4 of 10 goals needed a clarifying
  question. That is a feature of the design, not a weakness of the models, but it means a
  silent demonstration is often ambiguous.
- **One table, one arm, simple objects.** Nothing here has touched a real robot yet.

## Built during the hackathon

Everything in this repository was written between Oct 27 and Nov 4, 2026, by one
person. Dates are Bangkok time.

| Date | What |
| --- | --- |
| Oct 27 | SAM 3.1 video tracking sanity check, Muse Spark auto-naming test |
| Oct 28 | Ten clips recorded; tracking to table-plane positions; batch runner |
| Oct 29 | Events: rest intervals, hand-contact filtering, container snapping, rigid grouping, dedup |
| Oct 30 | Narration transcription; goal inference (prompt v1 to v3) |
| Oct 31 | MuJoCo simulator with IK, planner for five goal types, shuffled-scene generalization, LeRobot export |
| Nov 1 | Evaluation tables, no-narration ablation, project renamed to Grasp |
| Nov 2 | Web demo with live shuffle and upload |

`sam_api.py` is Meta's helper from the Model API cookbook (MIT licence), included
unchanged. The Franka Panda model is from MuJoCo Menagerie.

## Licence

MIT
