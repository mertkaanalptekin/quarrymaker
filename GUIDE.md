# Quarry Maker: guide for designers

This guide is for people who work with maps, terrain and 3D models, not with code. You never need
to open a terminal or edit a code file: everything happens in a simple app in your web browser.

## What you get

| File | What it is | Use it in |
|---|---|---|
| **Measured terrain** (`quarry_base.blend`) | The terrain exactly as measured by satellite radar (30 m grid), with the satellite photo on it. True scale, real coordinates. | Blender (with BlenderGIS) |
| **Quarry with calculated benches** (`quarry_terraced.blend`) | The same terrain, with the quarry's steps (benches) **calculated from the satellite photo**. | Blender |
| **Rhino files** (`.obj`, `.fbx`) | Both terrains as meshes, in metres, Z up. | Rhino |
| **Gaea package** | A square heightmap plus a short guide to set the right size and height in Gaea. | Gaea |
| **Reports** | What was done, the checks, and the known limits. | Reading |

**Measured vs calculated.** The measured terrain is the source of truth for scale and position.
The benches are an *interpretation*: a careful, checkable estimate made from the photo, not a
survey. The two files never mix, and both line up exactly in the same coordinates.

## Before the first use (once)

1. **Python.** Download and install it from <https://www.python.org/downloads/>.
   - **Windows:** on the first installer screen, tick **"Add python.exe to PATH"**.
2. **Blender.** Download and install it from <https://www.blender.org/download/>. Version 4.2 or
   newer works.
3. **BlenderGIS** (recommended). This lets Blender add basemaps and other map data to the model.
   1. Download the ZIP from <https://github.com/domlysz/BlenderGIS> (green **Code** button →
      **Download ZIP**).
   2. In Blender, go to **Edit → Preferences → Add-ons → Install…** and pick the ZIP.
   3. Tick the add-on to enable it.
4. **OpenTopography key** (free). The height data comes from OpenTopography.
   1. Create an account at <https://portal.opentopography.org>.
   2. Go to **My Account → Request API key**.
   3. Keep the key somewhere safe, for example in a password manager.
5. **Quarry Maker itself.**
   1. On <https://github.com/mertkaanalptekin/quarrymaker>, press the green **Code** button →
      **Download ZIP**.
   2. Unzip it somewhere easy to find, for example in Documents.

## Starting Quarry Maker

Double-click the start file in the Quarry Maker folder:

- **Mac:** `Start Quarry Maker.command`. The first time, macOS may warn that the file is from an
  unknown developer. **Right-click** the file and choose **Open**, then confirm.
- **Windows:** `Start Quarry Maker.bat`. If a blue "Windows protected your PC" box appears, click
  **More info → Run anyway**.

The first start takes a few minutes while it installs what it needs. After that it starts in
seconds.

A small window with text stays open, and your web browser opens the app. **Keep that small window
open while you work; closing it stops the app.** If the browser does not open by itself, copy the
link shown in the small window into your browser.

The app runs only on your computer. Nothing you enter is published online.

## The four steps in the app

### 1. Your quarry
1. Paste coordinates from Google Maps into the search box and press **Go there**. In Google Maps,
   right-click on the quarry and click the numbers at the top of the menu to copy them.
2. Press **Draw area**, then click one corner and the opposite corner of a rectangle around the
   quarry. Include some of the surroundings.
3. Optional: press **Trace pit outline**, click around the edge of the pit, then press **Finish
   outline**. Do this if step 7 later picks up the wrong area, for example bright fields next to
   the quarry.
4. Press **Save area**.

The box under the map shows the size and the coordinate system. The app chooses the right UTM zone
for your location (metres).

### 2. Settings
- **Blender.** Pick the Blender the app found, or choose **Other** and type its location.
- **OpenTopography key.** Paste your key and press **Use this key**. For safety the key is only kept
  while the app is open; it is never saved. You paste it again next time.
- **Bench height.** Quarry walls are cut as giant steps; the bench height is how tall one step is
  (usually 10–15 m).
  - **If you don't know it,** leave "calculate it from the image" selected.
  - **If you can find it,** the bench levels become more accurate. Ask the quarry operator, look in
    the permit documents, or measure it in Google Earth Pro: hover over one bench floor and then
    the next one up, and subtract the two elevations shown at the bottom right. Average a few.

Press **Save settings**.

### 3. Build the model
Press **Run next step**. After each step, look at its check images (click to enlarge). Each image
has a caption telling you what to look for. When they look right, press **Run next step** again.
Once you trust the process, you can use **Run all remaining steps**.

The steps:

| Step | What it does | What to check |
|---|---|---|
| Check setup | Python, Blender, key | Everything green |
| 1 | Downloads height data | The quarry is inside the shaded relief |
| 2 | Downloads the satellite photo | It is the right place |
| 3 | Aligns photo and heights | Relief and photo line up |
| 4 | Builds the Blender terrain | The terrain looks right |
| 5 | Makes the Gaea package | The red box covers the area |
| 6 | Measures and checks everything | All checks pass |
| 7 | Calculates the benches | The bench evidence map: green where benches were found. The benches look plausible. |

Downloads are kept, so running again is quick.

### 4. Your results
Press **Open in Blender** or **Open folder**.

## Using the results

**Blender**
- Open the file. The scene is in metres at true scale.
- The scene carries BlenderGIS georeferencing, so BlenderGIS basemaps and imports land in the
  right place.
- In the benches file, select `TERRAIN_terraced` and switch its material to
  `MAT_terrace_confidence`. That shows where the calculation is sure (green) and unsure (red).

**Rhino**
1. Set the document units to **Meters** before importing.
2. Import the `.obj`. In the import options, turn **off** "Map OBJ Y to Rhino Z".
3. The model sits near 0,0. To place it at real map coordinates, see `rhino_import_notes.md`.

**Gaea**
- Follow `README_gaea.md` in the Gaea folder. It tells you exactly which width and height to enter.

## About step 7 (benches)

Step 7 first finds the **quarry area** in the photo. This is bare rock and dumps with relief, kept
separate from fields, greenhouses and villages. If it includes something that isn't quarry (for
example a factory), trace the quarry outline yourself in step 1.

It then traces the **bench lines** in the photo: the long light and dark edges that bench faces and
their shadows draw. Each line becomes a step cut into the terrain, and the ground between lines
becomes a flat bench. The **height of each step comes from the elevation data** (how much the ground
drops there), so hills and low points stay as measured. The image `07_bench_lines.png` shows the
traced lines in red. These benches are an **illustration**: their positions follow the photo, but
where the quarry was dug deeper after the elevation data was captured (2011–2015), the steps come
out shallower than in reality.

Where the elevation data itself confirms benches, those parts are built from it instead (green in
`07_benches_measured_and_traced.png`); the parts cut along photo lines are orange.


Step 7 looks at your pit **region by region**, in squares of about 400 m. A big mining complex has
several pits and waste dumps, each with its own bench levels, and some parts may have been dug
after the elevation data was captured (2011–2015).

- **Green regions:** benches were found and are built in the model.
- **Amber regions:** something bench-like was seen, but it was not convincing enough.
- **Red regions:** no bench pattern.

Wherever there is no green, the model keeps the measured terrain; the program never invents benches.
Each pit or dump gets its own bench height and floor elevations, listed in the report.

If step 7 stops with "no consistent bench pattern", your other results (steps 1–6) are complete
and usable. Look at the bench evidence image, then:
- if you know the bench height, enter it in step 2 and run step 7 again;
- if most of the quarry was dug after 2015, the elevation data is simply too old to show its
  benches; untick "Calculate the benches".

## Plan linework (step 8)

Step 8 writes `quarry_plan.dxf` (open it in Rhino with units set to Meters) with these layers:
- **QUARRY_OUTLINE:** the quarry area.
- **BENCH_LINES:** the bench edges traced in the photo, as 3D lines at their elevation. Use
  *ProjectToCPlane* in Rhino for a flat plan. Shadow edges are left out; the sun direction is set in
  `config.yaml` (`sun_azimuth_deg`, 135 = southeast).
- **ROADS:** road centrelines.
- **AGRICULTURE:** farmland outlines.
- **BLOCK_YARDS:** block stone depositing and staging areas.

The coordinates match the Rhino OBJ (local metres; see README_plan.md for the origin). All
layers are traced automatically and are interpretations, not a survey.

## Quarry expansion over time (step 9)

Step 9 looks through Esri's archive of older satellite maps (World Imagery Wayback, 2014 onwards).
It keeps each different photo of your site, finds the quarry area on each in the same way, and
reports the area per date. It also draws a map of where the quarry grew, and writes
`quarry_expansion.dxf` with one outline layer per date.

The dates are when each photo appeared in Esri's map; the photo itself was taken somewhat
earlier. Photos differ in season and colour, so compare outlines by eye as well as by numbers.

## If something goes wrong

The app shows a red box with what stopped and **what to do**. The common ones:

| Message | What to do |
|---|---|
| Key is not set | Paste your OpenTopography key in step 2 and press "Use this key". |
| Area is not set | Draw the area in step 1 and press "Save area". |
| Blender not found | Pick or type the Blender location in step 2. |
| Area too large | Draw a smaller rectangle. |
| No quarry pit was found | Trace the pit outline in step 1. |
| No consistent bench pattern | No part of the pit showed clear benches. See "About step 7" above. Steps 1–6 are still complete. |

If you are stuck, send the red message and a screenshot of the check images to your helper. Never
send your key.

## Versions

When you change the area or a setting and run again, the results go to a **new numbered folder**
(`v002`, `v003`, …). Earlier results are never overwritten, so you can always go back.

## Words used

- **DEM**: elevation data on a grid; here one height every 30 m.
- **Ortho / satellite photo**: the image draped on the terrain.
- **UTM / EPSG:326xx**: a map projection in metres. The app chooses the right zone for you.
- **Origin**: the point that becomes 0,0 in Blender and Rhino, chosen near the corner of your area.
  Real map coordinates = model coordinates + origin.
- **Bench**: one step of the quarry wall. It has a flat **floor** and a steep **face**.
- **Hillshade**: a grey image of the terrain lit from one side, so relief is easy to see.
- **Interpreted**: estimated from the photo, as opposed to measured.
