# Atlas Framework

<br>

## Part 1: The Reference Space 

```python
glasser_img = nib.load("glasser_360_MNI152NLin6Asym.nii.gz")
```

`nib.load` reads a file from disk. The result is a Python object with three things inside:

### The Data (`.get_fdata()`)

A 3D NumPy array of shape `(182, 218, 182)`. That's 182 voxels in the X direction (left-right), 218 in Y (posterior-anterior), 182 in Z (inferior-superior). Total: 7,221,032 voxels. Each voxel is an integer.

Here's a real slice from your actual Glasser data at Z ≈ 75 (around the precentral gyrus):

```
X=85 to 95, Y=108 to 118, Z=75:

         Y=108  109  110  111  112  113  114  115  116  117  118
X=85  │   0    0    0    0    0    0    0    0    0    0    0
X=86  │   0    0    0    0    0    0    0    0    0    0    0
X=87  │   0    0    0    0    0    0    0    0    0    0    0
X=88  │   0    0    0    0    8    8    8    0    0    0    0    ← label 8 appears
X=89  │   0    0    0    8    8    8    8    8    0    0    0
X=90  │   0    0    8    8    8    8    8    8    8    0    0
X=91  │   0    0    8    8    8    8    8    8    8    0    0    ← center of M1
X=92  │   0    0    0    8    8    8    8    8    0    0    0
X=93  │   0    0    0    0    8    8    8    0    0    0    0
X=94  │   0    0    0    0    0    0    0    0    0    0    0
X=95  │   0    0    0    0    0    0    0    0    0    0    0
```

This is a cross-section through left M1. The 8s form an oval shape — that's the precentral gyrus. Each 8 is a 1mm³ cube of tissue. There are ~9,764 such cubes for left M1 total across all slices.

### The Affine Matrix (`.affine`)

A 4×4 matrix that converts voxel indices to MNI coordinates:

```python
ref_affine = [[  1.0,   0.0,   0.0, -91.0],
              [  0.0,   1.0,   0.0, -127.0],
              [  0.0,   0.0,   1.0,  -73.0],
              [  0.0,   0.0,   0.0,    1.0]]
```

The diagonal `[1.0, 1.0, 1.0]` means each voxel is 1mm. The last column `[-91, -127, -73]` is the translation — where voxel (0,0,0) sits in MNI space.

To convert voxel `[88, 112, 75]` to MNI:
```
[88, 112, 75, 1] × affine = [88×1 + 0 + 0 + 1×(-91), ...]
                           = [-3, -15, 2]
```

### The Header (`.header`)

Metadata: voxel dimensions, data type, coordinate system codes (sform/qform tell software this is MNI space).

```python
voxel_size = np.abs(np.diag(ref_affine)[:3])  # [1.0, 1.0, 1.0]
```

<br>

## Part 2: Glasser + Tian — The Offset Strategy

### Why Not Just Add the Labels?

Glasser uses labels 1-360. Tian uses labels 1-54. If we simply added the two arrays, voxel with Glasser label 8 and voxel with Tian label 8 would both be... 8. We'd lose which is which.

### The Solution: Label Offsetting

Tian gets shifted by +360:

```
Tian label 1  → 361
Tian label 2  → 362
...
Tian label 54 → 414
```

Now 1-360 are exclusively Glasser, 361-414 are exclusively Tian. No collision.

### Resampling Tian from 2mm to 1mm

Tian's original shape is `(91, 109, 91)` at 2mm. Glasser is `(182, 218, 182)` at 1mm. Each Tian voxel becomes 8 Glasser voxels (2×2×2 in each direction):

```
Tian 2mm voxel:          After resampling to 1mm:
┌─────────┐              ┌────┬────┐
│  label  │              │ 3  │ 3  │
│    3    │      →       ├────┼────┤
│         │              │ 3  │ 3  │
└─────────┘              └────┴────┘
```

`interpolation='nearest'` is critical here. It copies the label value to all 8 sub-voxels. If we used `interpolation='linear'`, the boundary between label 3 and label 4 would produce values like 3.2, 3.7 — meaningless for a label.

### The Merge

```python
gt_data = np.where(tian_data > 0, tian_data, glasser_data)
```

This checks every voxel: "Does Tian have a label here?" If yes, use Tian's label. If no, use Glasser's. Since Glasser covers cortex and Tian covers subcortex, they occupy different voxels. The `where` statement handles the rare boundary overlap.

<br>

## Part 3: Nettekoven — A Different Shape, Same Principle

Nettekoven is `(153, 103, 84)` at 1mm. It only covers the cerebellum, so it's a smaller grid embedded in a different corner of MNI space. When we resample to Glasser's `(182, 218, 182)`, all voxels outside the cerebellum become 0 automatically.

```
Original Nettekoven grid:        Resampled to Glasser grid:
┌────────────────────┐           ┌────────────────────┐
│  cerebellum only    │           │ 0 0 0 0 0 0 0 ... │
│  153×103×84        │     →     │ 0 0 0 0 0 0 0 ... │
│                    │           │ 0 0 cerebellum ... │
└────────────────────┘           │ 0 0 cerebellum ... │
                                 └────────────────────┘
```

The offset is +414, so Nettekoven labels 1-32 become 415-446.

<br>

## Part 4: STN Spheres

### Step A: MNI → Voxel

The inverse affine converts MNI coordinates to voxel indices:

```python
inv_affine = np.linalg.inv(ref_affine)
# = [[ 1,  0,  0,  91],
#    [ 0,  1,  0, 127],
#    [ 0,  0,  1,  73],
#    [ 0,  0,  0,   1]]

vox = inverse_affine × [-11.89, -14.51, -6.40, 1]
# = [-11.89 + 91, -14.51 + 127, -6.40 + 73]
# = [79.11, 112.49, 66.60]
# → rounded to [79, 112, 67]
```

So the left STN sits at voxel coordinates (79, 112, 67) in the grid.

### Step B: The Bounding Box

Rather than checking all 7.2 million voxels, we only check a small box around the STN:

```python
radius_vox = ceil(5mm / 1mm) + 1 = 6
x_min = 79 - 6 = 73,  x_max = 79 + 6 + 1 = 86  (13 voxels)
y_min = 112 - 6 = 106, y_max = 112 + 6 + 1 = 119 (13 voxels)
z_min = 67 - 6 = 61,  z_max = 67 + 6 + 1 = 74  (13 voxels)
```

A 13×13×13 = 2,197 voxel box. We only compute distances for these.

### Step C: The Distance Grid

```python
xx, yy, zz = np.mgrid[73:86, 106:119, 61:74]
```

This creates three 3D arrays. `xx[i,j,k]` is the X coordinate of voxel (i,j,k) in the bounding box. Same for `yy` and `zz`.

For voxel at local position (0,0,0) which is grid position (73, 106, 61):
```
distance = sqrt((73-79)² + (106-112)² + (61-67)²)
         = sqrt(36 + 36 + 36)
         = sqrt(108)
         = 10.4 mm  → outside 5mm sphere
```

For voxel at local position (6,6,6) which is grid position (79, 112, 67):
```
distance = sqrt((79-79)² + (112-112)² + (67-67)²) = 0.0 mm → center of sphere
```

### Step D: The Sphere Mask

```python
sphere_mask = distances <= 5.0
# True for ~515 voxels that are within 5mm of the STN center
```

Visualizing a 2D cross-section through the sphere:

```
Z = 67 (center slice):
         Y=106 107 108 109 110 111 112 113 114 115 116 117 118
X=73  │   .   .   .   .   .   .   .   .   .   .   .   .   .
X=74  │   .   .   .   .   .   .   .   .   .   .   .   .   .
X=75  │   .   .   .   .   .  447 447 447  .   .   .   .   .
X=76  │   .   .   .  447 447 447 447 447 447 447  .   .   .
X=77  │   .   .  447 447 447 447 447 447 447 447 447  .   .
X=78  │   .  447 447 447 447 447 447 447 447 447 447 447  .
X=79  │   .  447 447 447 447 447  ●  447 447 447 447 447  .   ← center
X=80  │   .  447 447 447 447 447 447 447 447 447 447 447  .
X=81  │   .   .  447 447 447 447 447 447 447 447 447  .   .
X=82  │   .   .   .  447 447 447 447 447 447 447  .   .   .
X=83  │   .   .   .   .   .  447 447 447  .   .   .   .   .
X=84  │   .   .   .   .   .   .   .   .   .   .   .   .   .
X=85  │   .   .   .   .   .   .   .   .   .   .   .   .   .
```

The ● marks voxel (79, 112, 67) — the exact MNI coordinate. The 447s form a circle of radius 5 voxels. Because each voxel is 1mm, this is exactly a 5mm radius sphere.

<br>

## Part 5: Merge & Validate

### Overlap Detection

```python
overlap_gt_nk = np.sum((gt_data > 0) & (nk_data > 0))
```

`(gt_data > 0) & (nk_data > 0)` is True only where BOTH arrays have non-zero values. `np.sum()` counts them. We found:

- GT-Nettekoven: 499 voxels (0.007% of brain) — boundary between brainstem and cerebellum
- GT-STN: 22 voxels — Tian subcortex borders STN territory
- Nettekoven-STN: 0 voxels — anatomically impossible, correctly zero

### Priority Order

```python
combined = gt_data.copy()          # Base layer: Glasser + Tian
combined[nk_data > 0] = nk_data   # Nettekoven overwrites base
combined[stn_data > 0] = stn_data  # STN overwrites everything
```

For the 499 overlapping GT-Nettekoven voxels → Nettekoven wins (cerebellum stays cerebellar). For the 22 overlapping GT-STN voxels → STN wins (your explicit sphere over Tian's implicit coverage).

After this, every voxel has exactly one integer. No voxel has two labels.

### Validation

```python
expected = set(range(1, 449))  # {1, 2, 3, ..., 448}
missing = expected - unique_labels
extra = unique_labels - expected
```

We check: do we have exactly labels 1 through 448 with no gaps and no extras? If `missing` or `extra` is non-empty, something went wrong.

<br>

## Part 6: Save 

```python
combined_img = nib.Nifti1Image(combined, ref_affine, glasser_img.header)
```

`Nifti1Image` takes three things:
1. The 3D array of labels (7.2 million integers)
2. The affine matrix (so MNI coordinates work)
3. The header (so other software knows voxel size, data type, etc.)

The result is a single `.nii.gz` file. When you later call `nib.load("CIMT_448ROIs_atlas.nii.gz")`, you get back the exact same 3D array with the exact same coordinate transform. The CSV maps each integer 1-448 to a human-readable name, hemisphere, and functional system.

<br>

## Diagram

```
Input files:                         Processing:                      Output:
                                                             
Glasser (360 labels)  ─┐                                   
                       ├─→ Keep as-is ─────────────┐        
Tian (54 labels)  ────┤                             │        
                       └─→ Resample + offset +360 ─┘        
                                                      ├─→ gt_data (1-414)
Nettekoven (32 labels) ──→ Resample + offset +414 ───┤        
                                                      ├─→ combined (1-448)
STN coords (2 points)  ──→ Sphere rasterize ─────────┘        
                                                             
                                                             ↓
                                              CIMT_448ROIs_atlas.nii.gz
                                              (182, 218, 182) @ 1mm
                                              Every voxel: 0 or 1-448
                                              No overlaps, no gaps
```
