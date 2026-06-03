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

REGION_MAP = {
    "Czech": "Česko",
    "Hungarian": "Magyarország",
    "English": "United States",
    "Simplified Chinese": "中国",  
    "Traditional Chinese": "台灣", 
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

def click_list_item_safe(d, text_to_find):
    elements = d(textContains=text_to_find)
    if not elements.exists:
        return False
        
    time.sleep(0.3) # SPEEDUP
    
    for i in range(elements.count):
        try:
            el = elements[i]
            info = el.info
            cls = str(info.get('className', '')).lower()
            res = str(info.get('resourceName', '') or info.get('resourceId', '')).lower()
            
            if 'edittext' in cls: continue
            if 'search' in res: continue
            
            bounds = info.get('bounds', {})
            if bounds:
                h = bounds.get('bottom', 0) - bounds.get('top', 0)
                if h < 10: continue
                
            el.click(timeout=1)
            return True
        except Exception:
            continue
            
    try:
        tv_elements = d(textContains=text_to_find, className="android.widget.TextView")
        if tv_elements.exists and tv_elements.count > 0:
            tv_elements[-1].click(timeout=1)
            return True
        if elements.count > 0:
            elements[-1].click(timeout=1)
            return True
    except Exception:
        pass
    return False

def change_language(d, target_lang_name):
    print(f"Opening Locale Settings on {d.serial}...")
    d.shell("am start -a android.settings.LOCALE_SETTINGS")
    time.sleep(1.0) # SPEEDUP: Was 2.0

    target_ui_name = LANGUAGE_MAP.get(target_lang_name, target_lang_name)
    target_region = REGION_MAP.get(target_lang_name, "")

    print(f"Scanning main language list for '{target_ui_name}'...")
    found_in_main = False
    last_page = ""
    stuck_count = 0
    
    while True:
        if d(textContains=target_ui_name).exists:
            found_in_main = True
            print(f"-> Found '{target_ui_name}' in the main list.")
            break
            
        if d(textContains="Add a language").exists or d(resourceIdMatches=".*add_language.*").exists:
            break
            
        current_page = d.dump_hierarchy()
        if current_page == last_page:
            stuck_count += 1
            if stuck_count >= 2: break 
        else:
            stuck_count = 0
            
        last_page = current_page
        d.swipe(0.5, 0.8, 0.5, 0.3, duration=0.1) # SPEEDUP
        time.sleep(0.2) # SPEEDUP

    if not found_in_main:
        print(f"'{target_ui_name}' not in main list. Adding via Search...")
        
        add_btn = d(resourceIdMatches=".*add_language.*")
        if not add_btn.exists:
            add_btn = d(textContains="Add a language")
        
        if add_btn.exists:
            add_btn.click()
            time.sleep(0.5) # SPEEDUP: Was 1.5
            
            search_clicked = False
            for search_id in [".*action_search.*", ".*menu_search.*", ".*search_button.*", ".*search.*"]:
                btn = d(resourceIdMatches=search_id, clickable=True)
                if btn.exists:
                    btn[0].click()
                    search_clicked = True
                    break
            
            if not search_clicked:
                btn = d(descriptionMatches="(?i).*search.*", clickable=True)
                if btn.exists:
                    btn[0].click()
                    search_clicked = True
                    
            if search_clicked:
                time.sleep(0.3) # SPEEDUP
                edit_text = d(className="android.widget.EditText")
                if edit_text.exists:
                    edit_text[0].set_text(target_ui_name)
                else:
                    d.send_keys(target_ui_name)
                time.sleep(0.5) # SPEEDUP
            else:
                d.send_keys(target_ui_name)
                time.sleep(0.5)

            found_langs = d(textContains=target_ui_name)
            if found_langs.exists:
                click_list_item_safe(d, target_ui_name)
                time.sleep(0.5) # SPEEDUP
                
                if not d(resourceIdMatches=".*add_language.*").exists and not d(textContains="Add a language").exists:
                    region_clicked = False
                    if target_region:
                        reg_search_clicked = False
                        for search_id in [".*action_search.*", ".*menu_search.*", ".*search_button.*", ".*search.*"]:
                            btn = d(resourceIdMatches=search_id, clickable=True)
                            if btn.exists:
                                btn[0].click()
                                reg_search_clicked = True
                                break
                                
                        if not reg_search_clicked:
                            btn = d(descriptionMatches="(?i).*search.*", clickable=True)
                            if btn.exists:
                                btn[0].click()
                                reg_search_clicked = True
                                
                        if reg_search_clicked:
                            time.sleep(0.3) # SPEEDUP
                            edit_text = d(className="android.widget.EditText")
                            if edit_text.exists:
                                edit_text[0].set_text(target_region)
                            else:
                                d.send_keys(target_region)
                            time.sleep(0.5) # SPEEDUP
                            
                        if click_list_item_safe(d, target_region):
                            region_clicked = True
                            
                    if not region_clicked:
                        region_option = d(resourceId="android:id/text1")
                        if not region_option.exists:
                            region_option = d(className="android.widget.TextView", clickable=True)
                        if region_option.exists:
                            try:
                                clicked_fb = False
                                for i in range(region_option.count):
                                    item = region_option[i]
                                    if item.info.get('className') != 'android.widget.EditText':
                                        item.click(timeout=1)
                                        clicked_fb = True
                                        break
                                if not clicked_fb:
                                    region_option.click(timeout=1)
                            except Exception: pass
                    time.sleep(0.5) # SPEEDUP
            else:
                return False
        else:
            return False

    print("Scrolling back to the top of the list...")
    last_page = ""
    while True:
        d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.05) # SPEEDUP
        time.sleep(0.1) # SPEEDUP
        current_page = d.dump_hierarchy()
        if current_page == last_page: break
        last_page = current_page

    last_sy = -1
    stuck_count = 0
    while True:
        lang_elem = d(textContains=target_ui_name)
        if not lang_elem.exists:
            d.swipe(0.5, 0.8, 0.5, 0.4, duration=0.1)
            time.sleep(0.2)
            continue
            
        try:
            sx, sy = lang_elem[0].center()
            if last_sy != -1 and abs(last_sy - sy) < 20: stuck_count += 1
            else: stuck_count = 0
            last_sy = sy
            
            if stuck_count >= 2 or sy < 200:
                print(f"Verified: '{target_ui_name}' is currently active at the top (Y={sy}).")
                time.sleep(1.0) # SPEEDUP: Was 2.0
                return True
                
            d.drag(sx, sy, sx, 50, duration=0.5) # SPEEDUP
            time.sleep(0.5) # SPEEDUP
            d.swipe(0.5, 0.3, 0.5, 0.8, duration=0.1)
            time.sleep(0.2)
        except Exception as e:
            time.sleep(0.5)

def main():
    parser = argparse.ArgumentParser(description="Change Android device language via UI automation.")
    parser.add_argument("--serial", help="Serial number of the device")
    parser.add_argument("--language", required=True, help="Language to change to")
    parser.add_argument("--list", action="store_true", help="List connected devices")

    args = parser.parse_args()

    if args.list:
        devices = adb.device_list()
        for d in devices: print(f" - {d.serial}")
        return

    d = get_device(args.serial)
    if d: change_language(d, args.language)

if __name__ == "__main__":
    main()