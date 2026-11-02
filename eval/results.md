## Table 1: perception (Muse Spark naming + SAM 3.1 tracking)

| Clip | Objects (truth) | Named | Missed | Spurious | Raw tracks | After merge | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c01 | 6 | 6 | - | - | 8 | 6 | 100% |
| c02 | 4 | 4 | - | - | 6 | 4 | 99% |
| c03 | 2 | 2 | - | - | 3 | 2 | 100% |
| c04 | 3 | 3 | - | - | 3 | 3 | 100% |
| c05 | 2 | 2 | - | - | 2 | 2 | 99% |
| c06 | 6 | 6 | - | white paper#1 | 7 | 7 | 99% |
| c07 | 5 | 5 | - | - | 5 | 5 | 100% |
| c08 | 5 | 5 | - | - | 5 | 5 | 100% |
| c09 | 5 | 4 | silver keychain | - | 4 | 4 | 100% |
| c10 | 4 | 4 | - | - | 5 | 4 | 98% |

Naming recall: **98%** (41/42 objects). Coverage = frames tracked while the object was in view. Raw tracks > after merge means SAM found the same object under two names; the merge step fixed it.

## Table 2: events (what moved, what was attached)

| Clip | Moved (truth) | Moved correct | False moved | Missed moved | Static correct | Events | Attachment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c01 | 3 | 3 | - | - | 3 | 3 | - |
| c02 | 4 | 4 | - | - | 0 | 4 | - |
| c03 | 1 | 1 | - | - | 0 | 1 | - |
| c04 | 2 | 2 | - | - | 1 | 1 | ✓ |
| c05 | 2 | 2 | - | - | 0 | 2 | - |
| c06 | 5 | 4 | - | - | 1 | 4 | - |
| c07 | 1 | 1 | - | - | 4 | 1 | - |
| c08 | 2 | 2 | - | - | 3 | 1 | ✓ |
| c09 | 1 | 1 | - | - | 3 | 1 | - |
| c10 | 3 | 3 | - | - | 1 | 3 | - |

Moved-object recall: **96%** (23/24). Attachment column: whether the keychain was grouped with the wallet where the truth says it should be.

## Table 3: goal inference (Muse Spark), with and without narration

| Clip | Truth | With narration | Type | Scope | Order | Asked? | Without narration | Type | Scope | Asked? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c01 | `put_all(pencil, cup)` | `put_all(pencil, white container#0)` | ✓ | ✓ | ✓ | no | `put_all(pencil, white container#0)` | ✓ | ✓ | no |
| c02 | `arrange(objects, row)` | `arrange(object, line)` | ✓ | ✓ | ✓ | no | `arrange(object, row)` | ✓ | ✗ | no |
| c03 | `stack(small eraser, big eraser)` | `stack(small blue eraser#1, small blue eraser#0)` | ✓ | ✓ | ✓ | no | `stack(small blue eraser#1, small blue eraser#0)` | ✓ | ✓ | no |
| c04 | `move(black wallet, next_to, pencil)` | `move(black wallet#0, next to, brown pencil#0)` | ✓ | ✓ | ✓ | no | `move(black wallet#0, to)` | ✓ | ✓ | yes |
| c05 | `swap(scissor, eye glasses)` | `swap(white scissors#0, black glasses#0)` | ✓ | ✓ | ✓ | no | `swap(white scissors#0, black glasses#0)` | ✓ | ✓ | no |
| c06 | `put_all(object, cup)` | `put_all(object, white bowl#0)` | ✓ | ✗ | ✓ | no | `put_all(tool, white bowl#0)` | ✓ | ✓ | yes |
| c07 | `move(blue eraser, leftmost, table)` | `move(blue eraser#0, leftmost)` | ✓ | ✓ | ✓ | no | `move(blue eraser#0, left, black pencil#0)` | ✓ | ✓ | yes |
| c08 | `move(black wallet, left, table)` | `move(black wallet#0, left)` | ✓ | ✓ | ✓ | no | `move(black wallet#0, to)` | ✓ | ✓ | yes |
| c09 | `move(black pencil, to, position)` | `move(black pencil#0, to)` | ✓ | ✓ | ✓ | no | `move(black pencil#0, next to, black wallet#0)` | ✓ | ✓ | no |
| c10 | `put_all(pencil, cup)` | `put_all(pencil, white cup#0)` | ✓ | ✓ | ✓ | no | `put_all(pencil, white cup#0)` | ✓ | ✓ | no |

With narration: type **100%**, scope **90%**, clarifying question needed on **0/10** clips. Without narration: type **100%**, scope **90%**, clarifying question needed on **4/10** clips. Type and scope are scored on the top candidate. Narration mostly buys certainty: without it the model offers two readings and has to ask.

## Table 4: teach once, adapt to new scenes (simulation)

| Clip | Goal | Original: goal / copy | Shuffled scenes | Goal plan | Copy baseline | Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| c01 | `put_all(pencil, white container#0)` | ✓ / ✓ | 10 | 100% | 0% | 11 |
| c02 | `arrange(object, line)` | ✓ / ✓ | 10 | 100% | 0% | 11 |
| c03 | `stack(small blue eraser#1, small blue eraser#0)` | ✓ / ✓ | 10 | 100% | 0% | 11 |
| c04 | `move(black wallet#0, next to, brown pencil#0)` | ✓ / ✓ | 10 | 100% | 0% | 11 |
| c05 | `swap(white scissors#0, black glasses#0)` | ✓ / ✗ | 10 | 90% | 0% | 10 |
| c06 | `put_all(object, white bowl#0)` | ✓ / ✓ | 10 | 100% | 0% | 11 |
| c07 | `move(blue eraser#0, leftmost)` | ✓ / ✓ | 10 | 100% | 0% | 11 |
| c08 | `move(black wallet#0, left)` | ✓ / ✓ | 10 | 100% | 0% | 11 |
| c09 | `move(black pencil#0, to)` | ✓ / ✓ | 10 | 80% | 10% | 9 |
| c10 | `put_all(pencil, white cup#0)` | ✓ / ✓ | 10 | 100% | 0% | 11 |

Across 100 shuffled scenes: goal plan **97%**, copy baseline **1%**. **107 verified episodes from 10 phone demonstrations.** Shuffled scenes move every object; every second scene adds an extra object of the demonstrated kind and every third adds a distractor. Scenes where the goal already held before acting were rerolled. Success is judged against simulator state.
