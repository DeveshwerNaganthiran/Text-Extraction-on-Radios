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

    # 1. CHECK IF ALREADY IN MAIN LIST (Do a few short scrolls to check)
    found_in_main = False
    last_page = ""
    for _ in range(3):
        if d(textContains=target_ui_name).exists:
            found_in_main = True
            break
        current_page = d.dump_hierarchy()
        if current_page == last_page:
            break
        last_page = current_page
        d.swipe(0.5, 0.8, 0.5, 0.4, duration=0.2)
        time.sleep(0.5)

    # 2. IF NOT FOUND, ADD VIA SEARCH
    if not found_in_main:
        print(f"'{target_ui_name}' not in main list. Adding via Search...")
        
        # Find the Add button
        add_btn = d(resourceIdMatches=".*add_language.*")
        if not add_btn.exists:
            for _ in range(5):
                if add_btn.exists: break
                d.swipe(0.5, 0.8, 0.5, 0.3, duration=0.2)
                time.sleep(0.5)
        
        if add_btn.exists:
            add_btn.click()
            time.sleep(1.5)
            
            # FORCE SEARCH INSTEAD OF SCROLLING
            search_btn = d(resourceIdMatches=".*action_search.*")
            if search_btn.exists:
                search_btn.click()
                time.sleep(1)
                d.send_keys(target_ui_name)
                time.sleep(1.5)
            else:
                alt_search = d(descriptionMatches=".*[Ss]earch.*")
                if alt_search.exists:
                    alt_search.click()
                    time.sleep(1)
                    d.send_keys(target_ui_name)
                    time.sleep(1.5)

            # Click the language search result
            result = d(textContains=target_ui_name)
            if result.exists:
                try:
                    result[-1].click(timeout=3)
                except Exception as e:
                    print(f"Warning on search click: {e}")
                time.sleep(2.0)
                
                # =========================================================
                # NEW: AUTOMATED REGION / COUNTRY SELECTION INTERCEPT
                # =========================================================
                # If the '+ Add a language' button is still NOT visible, 
                # it means Android is forcing a region choice screen.
                if not d(resourceIdMatches=".*add_language.*").exists:
                    print("Detected region selection screen. Picking the first available region...")
                    
                    # Try to find the first clickable choice inside the list view
                    region_option = d(resourceId="android:id/text1")
                    if not region_option.exists:
                        region_option = d(className="android.widget.TextView", clickable=True)
                    if not region_option.exists:
                        region_option = d(className="android.widget.LinearLayout", clickable=True, instance=0)
                    
                    if region_option.exists:
                        try:
                            region_option.click(timeout=3)
                            print("Region chosen successfully.")
                        except Exception as e:
                            print(f"Warning clicking region: {e}")
                        time.sleep(2.5)
                # =========================================================
                
            else:
                print(f"Could not find '{target_ui_name}' in the Add menu search results.")
                return False
        else:
            print("Could not find the '+ Add a language' button.")
            return False

    # 3. FAST SCROLL TO ABSOLUTE TOP
    last_page = ""
    while True:
        d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.1) # Fast Scroll UP
        time.sleep(0.2)
        current_page = d.dump_hierarchy()
        if current_page == last_page:
            break # Reached the top
        last_page = current_page

    # 4. CONTINUOUS DRAGGING (Bounce-Back Detection)
    last_sy = -1
    stuck_count = 0
    
    while True:
        lang_elem = d(textContains=target_ui_name)
        if not lang_elem.exists:
            d.swipe(0.5, 0.8, 0.5, 0.4, duration=0.2) # Scroll down slowly
            time.sleep(0.5)
            continue
            
        try:
            sx, sy = lang_elem[-1].center()
            
            # Check if it bounced back to the exact same spot (hit the ceiling)
            if last_sy != -1 and abs(last_sy - sy) < 20:
                stuck_count += 1
            else:
                stuck_count = 0
                
            last_sy = sy
            
            # If stuck in same spot twice or very high up, it is in Slot #1
            if stuck_count >= 2 or sy < 200:
                print(f"Verified: '{target_ui_name}' is currently active at the top (Y={sy}).")
                return True
                
            print(f"Dragging '{target_ui_name}' from Y={sy} upwards...")
            d.drag(sx, sy, sx, 100, duration=0.4) # Drag it high up
            time.sleep(1.0)
            
            d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.2)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error during drag operation: {e}")
            time.sleep(1)

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