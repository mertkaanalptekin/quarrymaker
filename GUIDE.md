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
| 7 | Calculates the benches | The coloured pit area covers the quarry; the benches look plausible |

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

## If something goes wrong

The app shows a red box with what stopped and **what to do**. The common ones:

| Message | What to do |
|---|---|
| Key is not set | Paste your OpenTopography key in step 2 and press "Use this key". |
| Area is not set | Draw the area in step 1 and press "Save area". |
| Blender not found | Pick or type the Blender location in step 2. |
| Area too large | Draw a smaller rectangle. |
| No quarry pit was found | Trace the pit outline in step 1. |
| No consistent bench pattern | The benches are not clear in the photo. Enter the bench height in step 2, or trace the pit outline. |

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
