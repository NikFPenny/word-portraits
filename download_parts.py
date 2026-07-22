import requests
import json
from requests.auth import HTTPBasicAuth
import time
import os

# ---------------------------------------------------------------------------
# SECURITY NOTE: fill your keys in via environment variables, not literal
# strings in this file -- that way they never end up in a chat log, a git
# commit, or anywhere else this file gets copied to.
#
#   export ONSHAPE_ACCESS_KEY="your_access_key"
#   export ONSHAPE_SECRET_KEY="your_secret_key"
# ---------------------------------------------------------------------------

ACCESS_KEY = 'PTIcFNlK7nUH3ZKzoVJ8ezT2'
SECRET_KEY = 'MyxkXXLWgNhdWv8Jk1vqjOHGRV9JpejwFboz9p5It6HcqCLI'

AUTH = HTTPBasicAuth(ACCESS_KEY, SECRET_KEY)

### "back up wrapped spline" / Split By Face Index project ###
# -----------------------------------------------------------------------------------
# Fill these in from your Part Studio's URL:
#   https://cad.onshape.com/documents/{document_id}/w/{work_space_id}/e/{element_id}
#afaeaafa515523272055262d/w/39ec59fcf5975a9d09cc89ca/e/9d5df55f0920159867d76f75
document_id = 'afaeaafa515523272055262d'
work_space_id = '39ec59fcf5975a9d09cc89ca'
element_id = '9d5df55f0920159867d76f75'

FEATURE_NAME = "Split By Face Index"   # must match the feature's name in the tree
NUM_FACES = 71                         # loop from faceIndex 0 to NUM_FACES - 1
TARGET_PART_NAME = "Part 2"             # which resulting part to download each time
OUTPUT_DIR = "ShellExports"


def print_remaining_calls(response):
    """
    Print how many Onshape API calls are left in the current rate-limit window.
    """
    remaining = response.headers.get('X-Rate-Limit-Remaining')
    if remaining is not None:
        print(f"API calls remaining: {remaining}")


def get_feature_definition():
    """
    Retrieve the "Split By Face Index" feature's definition from Onshape,
    by matching on its name (not just grabbing features[0], since your
    Part Studio may have more than one feature in the tree).
    """
    url = (
        f"https://cad.onshape.com/api/v10/partstudios/d/{document_id}/w/{work_space_id}"
        f"/e/{element_id}/features?rollbackBarIndex=-1&includeGeometryIds=true&noSketchGeometry=false"
    )
    response = requests.get(url, auth=AUTH)
    print_remaining_calls(response)
    time.sleep(2)

    if response.status_code != 200:
        print(f"Failed to retrieve features: {response.status_code} - {response.text}")
        return None

    features = response.json()["features"]
    for feature in features:
        if feature.get("name") == FEATURE_NAME:
            return feature

    print(f"Could not find a feature named '{FEATURE_NAME}'. "
          f"Available names: {[f.get('name') for f in features]}")
    return None


def update_feature(feature_definition, index):
    """
    Set the 'faceIndex' parameter on the feature to `index` and push the
    update to Onshape, triggering a regeneration.
    """
    feature_data = feature_definition
    if not feature_data:
        return

    found = False
    for param in feature_data["parameters"]:
        if param["parameterId"] == "faceIndex":
            param["expression"] = f"{index}"
            found = True
    if not found:
        print("WARNING: no 'faceIndex' parameter found on this feature -- "
              "check FEATURE_NAME and the parameter's actual parameterId.")

    update_payload = {
        "feature": feature_data
    }
    feature_id = feature_data["featureId"]
    url = (
        f"https://cad.onshape.com/api/v10/partstudios/d/{document_id}/w/{work_space_id}"
        f"/e/{element_id}/features/featureid/{feature_id}"
    )
    response = requests.post(url, auth=AUTH, json=update_payload)
    print_remaining_calls(response)
    time.sleep(2)

    if response.status_code == 200:
        print("Feature updated successfully!")
    else:
        print(f"Failed to update feature: {response.status_code} - {response.text}")


def get_parts(feature_definition, index):
    """
    Update the feature to the given face index, then find and download
    the resulting shell part (TARGET_PART_NAME) as an STL file.
    """
    update_feature(feature_definition, index)

    ### STEP 2: GET PART LIST ###
    parts_response = requests.get(
        f"https://cad.onshape.com/api/v10/parts/d/{document_id}/w/{work_space_id}"
        f"?elementId={element_id}&withThumbnails=false&includePropertyDefaults=false",
        auth=AUTH
    )
    time.sleep(2)

    if parts_response.status_code != 200:
        print(f"Failed to get parts: {parts_response.text}")
        return

    parts_data = parts_response.json()

    ### STEP 3: FIND THE TARGET PART AND EXPORT IT AS STL ###
    target_part_id = None
    for part in parts_data:
        if part.get("name") == TARGET_PART_NAME:
            target_part_id = part["partId"]
            break

    if target_part_id is None:
        print(f"Could not find a part named '{TARGET_PART_NAME}' at face index {index} -- skipped.")
        return

    redirect_response = requests.get(
        f"https://cad.onshape.com/api/v10/parts/d/{document_id}/w/{work_space_id}"
        f"/e/{element_id}/partid/{target_part_id}/stl?mode=text&grouping=true&scale=1&units=millimeter",
        auth=AUTH,
        allow_redirects=False
    )
    time.sleep(2)
    stl_url = redirect_response.headers['Location']
    stl_response = requests.get(stl_url, auth=AUTH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = f"{OUTPUT_DIR}/shell_face_{index}.stl"

    if stl_response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(stl_response.content)
        print(f"Exported face {index}'s shell as STL to {file_path}.")
    else:
        print(f"Failed to export face {index}'s shell: {stl_response.text}")


if __name__ == "__main__":
    start = time.time()

    feature_definition = get_feature_definition()
    if feature_definition is None:
        print("Aborting -- could not find the target feature.")
    else:
        for i in range(NUM_FACES):
            # reuse the same in-memory feature definition across every
            # iteration -- update_feature() mutates it locally and posts
            # the change, so there's no need to re-fetch it from Onshape
            # each time (that was needlessly burning through this
            # endpoint's rate limit)
            get_parts(feature_definition, i)

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Total time elapsed: {minutes} minutes and {seconds} seconds.")