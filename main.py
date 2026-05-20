import argparse
import time
import uiautomator2 as u2
from adbutils import adb

# Mapping of requested languages to their display names in Android Locale Settings
LANGUAGE_MAP = {
    "Czech": "Čeština (Česko)",
    "Simplified Chinese": "简体中文",
    "Portuguese": "Português",
    "Spanish": "Español",
    "Polish": "Polski",
    "Italian": "Italiano",
    "Turkish": "Türkçe",
    "Hungarian": "Magyar (Magyarország)",
    "English": "English",
    "Japanese": "日本語",
    "Russian": "Русский",
    "French": "Français",
    "German": "Deutsch",
    "Korean": "한국어",
    "Traditional Chinese": "繁體中文"
}

def get_device(serial=None):
    if serial:
        return u2.connect(serial)
    else:
        devices = adb.device_list()
        if not devices:
            print("No devices found.")
            return None
        if len(devices) > 1:
            print("Multiple devices found. Please specify a serial number.")
            for d in devices:
                print(f" - {d.serial}")
            return None
        return u2.connect(devices[0].serial)

def change_language(d, target_lang_name):
    print(f"Opening Locale Settings on {d.serial}...")
    d.shell("am start -a android.settings.LOCALE_SETTINGS")
    time.sleep(2)

    target_ui_name = LANGUAGE_MAP.get(target_lang_name, target_lang_name)
    lang_elem = d(textContains=target_ui_name)

    # 1. Scroll down slowly to see if the language is already in the active list
    found_in_main = False
    for _ in range(5):
        if lang_elem.exists:
            found_in_main = True
            break
        d.swipe(0.5, 0.8, 0.5, 0.3, duration=0.2) # Scroll down
        time.sleep(0.5)

    # 2. If it's not in the list, scroll down to find the Add button
    if not found_in_main:
        print(f"'{target_ui_name}' not found in main list. Attempting to add it...")
        add_btn = d(resourceId="com.android.settings:id/add_language")
        found_add_btn = False
        
        for _ in range(5):
            if add_btn.exists:
                found_add_btn = True
                break
            d.swipe(0.5, 0.8, 0.5, 0.3, duration=0.2) # Scroll down
            time.sleep(0.5)
            
        if found_add_btn:
            add_btn.click()
            time.sleep(1.5)
            
            search_btn = d(resourceId="com.android.settings:id/action_search")
            if search_btn.exists:
                search_btn.click()
                time.sleep(1)
                d.send_keys(target_ui_name)
                time.sleep(1.5)
            
            result = d(textContains=target_ui_name)
            if result.exists:
                try: result[-1].click(timeout=3)
                except Exception as e: print(f"Warning on search click: {e}")
                time.sleep(1.5)
                
                if not d(resourceId="com.android.settings:id/add_language").exists:
                    try:
                        region_result = d(textContains=target_ui_name)
                        if region_result.exists: region_result[-1].click(timeout=3)
                    except: pass
                    time.sleep(1.5)
            else:
                print(f"Could not find '{target_ui_name}' in the Add menu.")
                return False
        else:
            print("Could not find the '+ Add a language' button.")
            return False

    # 3. Fast scroll to the top of the list
    for _ in range(5):
        d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.1) # Fast Scroll UP
        time.sleep(0.2)

    # 4. Iterative Dragging (Drag it up in steps)
    for attempt in range(4):
        lang_elem = d(textContains=target_ui_name)
        if not lang_elem.exists:
            d.swipe(0.5, 0.8, 0.5, 0.4, duration=0.2) # Scroll down slowly
            time.sleep(0.5)
            continue
            
        try:
            sx, sy = lang_elem[-1].center()
            if sy < 250: # Already near the top
                if sy > 150: # Minor adjustment to strictly reach Slot 1
                    d.drag(sx, sy, sx, 100, duration=0.2)
                    time.sleep(1)
                break
                
            print(f"Step {attempt+1}: Dragging '{target_ui_name}' from Y={sy} upwards...")
            d.drag(sx, sy, sx, 150, duration=0.5) # Drag it high up
            time.sleep(1.5)
            
            # Scroll up slightly so the list adjusts
            d.swipe(0.5, 0.3, 0.5, 0.7, duration=0.2)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error during drag operation: {e}")

    # 5. Verification step
    d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.2) # Ensure we are viewing the absolute top
    time.sleep(1)
    lang_elem = d(textContains=target_ui_name)
    if lang_elem.exists:
        try:
            # Since we dragged it up, it should be the top-most instance [0]
            _, sy = lang_elem[0].center()
            # Relaxed threshold: Just needs to be in the upper half of the screen
            if sy < 500:
                print(f"Verified: '{target_ui_name}' is currently active at the top (Y={sy}).")
                return True
            else:
                print(f"Verification Failed: '{target_ui_name}' is at Y={sy}, which is too low to be Slot 1.")
        except Exception as e: 
            print(f"Verification check error: {e}")
            pass
            
    print(f"Verification Failed: '{target_ui_name}' is not at the top of the list.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Change Android device language via UI automation.")
    parser.add_argument("--serial", help="Serial number of the device")
    parser.add_argument("--language", required=True, help="Language to change to")
    parser.add_argument("--list", action="store_true", help="List connected devices")

    args = parser.parse_args()

    if args.list:
        devices = adb.device_list()
        print("Connected devices:")
        for d in devices:
            print(f" - {d.serial}")
        return

    d = get_device(args.serial)
    if d:
        change_language(d, args.language)

if __name__ == "__main__":
    main()