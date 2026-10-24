set -euo pipefail
cd /workspace/v2d-track-2
cat > runs/i1screen/repeat_plan.json <<'PLAN'
{
 "repeats": 3,
 "candidates": [
  {
   "name": "I1-T2C-E4-s44-snack",
   "source": "I1-T2C-E4-s44-snack/T2C-E4-seed44",
   "arm": "T2C-E4-seed44",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-06_10-24-18_snack_box_pick_and_place_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-E4-s44-trash",
   "source": "I1-T2C-E4-s44-trash/T2C-E4-seed44",
   "arm": "T2C-E4-seed44",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-02-20_17-37-45_blue_trash_can_drag_007",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-E4-s44-corn",
   "source": "I1-T2C-E4-s44-corn/T2C-E4-seed44",
   "arm": "T2C-E4-seed44",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-23_18-10-01_corn_can_right_left_handover_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-contact-s44-snack",
   "source": "I1-T2C-contact-s44-snack/T2C-contact-seed44",
   "arm": "T2C-contact-seed44",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-06_10-24-18_snack_box_pick_and_place_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-contact-s44-trash",
   "source": "I1-T2C-contact-s44-trash/T2C-contact-seed44",
   "arm": "T2C-contact-seed44",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-02-20_17-37-45_blue_trash_can_drag_007",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-contact-s44-corn",
   "source": "I1-T2C-contact-s44-corn/T2C-contact-seed44",
   "arm": "T2C-contact-seed44",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-23_18-10-01_corn_can_right_left_handover_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-E4-s45-snack",
   "source": "I1-T2C-E4-s45-snack/T2C-E4-seed45",
   "arm": "T2C-E4-seed45",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-06_10-24-18_snack_box_pick_and_place_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-E4-s45-trash",
   "source": "I1-T2C-E4-s45-trash/T2C-E4-seed45",
   "arm": "T2C-E4-seed45",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-02-20_17-37-45_blue_trash_can_drag_007",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-E4-s45-corn",
   "source": "I1-T2C-E4-s45-corn/T2C-E4-seed45",
   "arm": "T2C-E4-seed45",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-23_18-10-01_corn_can_right_left_handover_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-contact-s45-snack",
   "source": "I1-T2C-contact-s45-snack/T2C-contact-seed45",
   "arm": "T2C-contact-seed45",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-06_10-24-18_snack_box_pick_and_place_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-contact-s45-trash",
   "source": "I1-T2C-contact-s45-trash/T2C-contact-seed45",
   "arm": "T2C-contact-seed45",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-02-20_17-37-45_blue_trash_can_drag_007",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  },
  {
   "name": "I1-T2C-contact-s45-corn",
   "source": "I1-T2C-contact-s45-corn/T2C-contact-seed45",
   "arm": "T2C-contact-seed45",
   "arms_file": "pilot_arms_i1.txt",
   "input": "raw_repaired",
   "sequence": "2026-03-23_18-10-01_corn_can_right_left_handover_01",
   "steps": [
    49999872
   ],
   "final_step": 49999872,
   "group": "i1"
  }
 ]
}
PLAN
.venv/bin/python -B scripts/prepare_repeats.py --plan runs/i1screen/repeat_plan.json --source-root runs/i1screen --out-root runs/i1screen/repeats
PILOT_EVALUATE_ONLY=1 JOBS_FILE=runs/i1screen/repeats/jobs-i1.txt scripts/test_day.sh "$(nvidia-smi -L | wc -l)" runs/i1screen/repeats/i1
