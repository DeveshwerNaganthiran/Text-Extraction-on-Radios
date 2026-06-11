import argparse
import time
import uiautomator2 as u2
from adbutils import adb

# Base native names for finding the language in the list or search results
LANGUAGE_MAP = {
    "Czech": "Čeština",
    "Simplified Chinese": "简体中文",
    "Portuguese": "Português",
    "Spanish": "Español",
    "Polish": "Polski",
    "Italian": "Italiano",
    "Turkish": "Türkçe",
    "Hungarian": "Magyar",
    "English": "English",
    "Japanese": "日本語",
    "Russian": "Русский",
    "French": "Français",
    "German": "Deutsch",
    "Korean": "한국어",
    "Traditional Chinese": "繁體中文"
}

# Preferred regions if Android prompts for one. 
REGION_MAP = {
    "Czech": "Česko",
    "Hungarian": "Magyarország",
    "English": "United States",
    "Simplified Chinese": "中国",  # China
    "Traditional Chinese": "台灣", # Taiwan
    "Spanish": "España",
    "French": "France",
    "German": "Deutschland",
    "Italian": "Italia",
    "Portuguese": "Brasil", 
    "Russian": "Россия"
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
    target_region = REGION_MAP.get(target_lang_name, "")

    # =====================================================================
    # 1. SCROLL MAIN LIST UNTIL BOTTOM TO FIND LANGUAGE
    # =====================================================================
    print(f"Scanning main language list for '{target_ui_name}'...")
    found_in_main = False
    
    # 1. Scroll to the very top first to ensure we don't miss anything above
    for _ in range(3):
        if d(textContains=target_ui_name).exists:
            found_in_main = True
            break
        d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.1)
        time.sleep(0.2)

    # 2. Scroll all the way down to check the rest of the list
    if not found_in_main:
        last_page = ""
        stuck_count = 0
        while True:
            if d(textContains=target_ui_name).exists:
                found_in_main = True
                break
                
            current_page = d.dump_hierarchy()
            if current_page == last_page:
                stuck_count += 1
                if stuck_count >= 2: break 
            else:
                stuck_count = 0
                
            last_page = current_page
            d.swipe(0.5, 0.8, 0.5, 0.3, duration=0.1) 
            time.sleep(0.2) 

    if found_in_main:
        print(f"-> Found '{target_ui_name}' in the main list.")

    if not found_in_main:
        print(f"'{target_ui_name}' not in main list. Adding via Search...")
        
        add_btn = d(resourceIdMatches=".*add_language.*")
        if not add_btn.exists:
            add_btn = d(textContains="Add a language")
        
        if add_btn.exists:
            add_btn.click()
            time.sleep(1.5)
            
            # --- STEP A: SCROLL TO FIND LANGUAGE ---
            print(f"-> Scrolling to find Language: '{target_ui_name}'")
            found_lang_to_add = False
            last_page = ""
            stuck_count = 0
            
            while True:
                lang_item = d(textContains=target_ui_name)
                if lang_item.exists:
                    time.sleep(0.5)
                    try:
                        if lang_item.count > 0:
                            lang_item[0].click(timeout=3)
                        else:
                            lang_item.click(timeout=3)
                        print(f"-> Language '{target_ui_name}' clicked successfully.")
                        found_lang_to_add = True
                    except Exception as e:
                        print(f"Warning clicking language: {e}")
                    break
                    
                current_page = d.dump_hierarchy()
                if current_page == last_page:
                    stuck_count += 1
                    if stuck_count >= 2:
                        break # Bottom reached
                else:
                    stuck_count = 0
                    
                last_page = current_page
                d.swipe(0.5, 0.8, 0.5, 0.3, duration=0.2) # Scroll down
                time.sleep(0.5)

            if not found_lang_to_add:
                print(f"Could not find '{target_ui_name}' in the Add menu.")
                return False
                
            time.sleep(2.0)
            
            # --- STEP B: SCROLL TO FIND REGION (If prompted) ---
            # If the "Add a language" button is still not visible, we are on a region screen
            if not d(resourceIdMatches=".*add_language.*").exists and not d(textContains="Add a language").exists:
                print("-> Region selection screen detected.")
                region_clicked = False
                
                if target_region:
                    print(f"-> Scrolling to find Region: '{target_region}'")
                    last_page = ""
                    stuck_count = 0
                    
                    while True:
                        reg_item = d(textContains=target_region)
                        if reg_item.exists:
                            time.sleep(0.5)
                            try:
                                if reg_item.count > 0:
                                    reg_item[0].click(timeout=3)
                                else:
                                    reg_item.click(timeout=3)
                                print(f"-> Region '{target_region}' chosen successfully.")
                                region_clicked = True
                            except Exception as e:
                                print(f"Warning clicking region: {e}")
                            break
                            
                        current_page = d.dump_hierarchy()
                        if current_page == last_page:
                            stuck_count += 1
                            if stuck_count >= 2:
                                break
                        else:
                            stuck_count = 0
                            
                        last_page = current_page
                        d.swipe(0.5, 0.8, 0.5, 0.3, duration=0.2) # Scroll down
                        time.sleep(0.5)
                        
                # Fallback to the first available choice if region wasn't found via scroll
                if not region_clicked:
                    print("-> Specific region not found or not provided. Picking the first available region...")
                    region_option = d(resourceId="android:id/text1")
                    if not region_option.exists:
                        region_option = d(className="android.widget.TextView", clickable=True)
                    if region_option.exists:
                        try:
                            if region_option.count > 0:
                                region_option[0].click(timeout=3)
                            else:
                                region_option.click(timeout=3)
                            print("-> First available region chosen successfully.")
                        except Exception as e:
                            print(f"Warning clicking fallback region: {e}")
                time.sleep(2.5)
        else:
            print("Could not find the '+ Add a language' button.")
            return False

    # =====================================================================
    # 3. FAST SCROLL TO ABSOLUTE TOP
    # =====================================================================
    print("Scrolling back to the top of the main list...")
    last_page = ""
    while True:
        d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.1) # Fast Scroll UP
        time.sleep(0.2)
        current_page = d.dump_hierarchy()
        if current_page == last_page:
            break # Reached the top
        last_page = current_page

    # =====================================================================
    # 4. CONTINUOUS DRAGGING (Bounce-Back Detection)
    # =====================================================================
    last_sy = -1
    stuck_count = 0
    
    while True:
        lang_elem = d(textContains=target_ui_name)
        if not lang_elem.exists:
            d.swipe(0.5, 0.8, 0.5, 0.4, duration=0.2) # Scroll down slowly
            time.sleep(0.5)
            continue
            
        try:
            # Grab the very first match on the screen [0]
            sx, sy = lang_elem[0].center()
            
            # Check if it bounced back to the exact same spot (hit the ceiling)
            if last_sy != -1 and abs(last_sy - sy) < 20:
                stuck_count += 1
            else:
                stuck_count = 0
                
            last_sy = sy
            
            # If stuck in same spot twice or very high up, it is in Slot #1
            if stuck_count >= 2 or sy < 200:
                print(f"Verified: '{target_ui_name}' is currently active at the top (Y={sy}).")
                time.sleep(2.0) # Ensure OS fully updates locale globally
                return True
                
            print(f"Dragging '{target_ui_name}' from Y={sy} upwards...")
            d.drag(sx, sy, sx, 50, duration=1.0) # Drag it high up
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