import itertools

# The 10 display styles you provided
styles = ["0038", "0039", "0040", "0041", "0050", "0051", "0052", "0053", "0054", "0055"]
style_cycle = itertools.cycle(styles)
# The 733 end strings from your list
strings = [
    "12", "13", "18", "21", "22", "40", "67", "85", "86", "90", "91", "143", "153", "177", "192", "206", "635", "907", "947", "948", "1124", "1125", "1131", "1143", "1144", "1145", "1146", "1147", "1148", "1179", "1184", "1185", "1186", "1187", "1189", "1190", "1191", "1192", "1193", "1194", "1195", "1196", "1197", "1199", "1200", "1201", "1202", "1203", "1204", "1205", "1206", "1207", "1208", "1209", "1213", "1214", "1220", "1227", "1228", "1229", "1230", "1231", "1232", "1233", "1240", "1241", "1246", "1250", "1251", "1252", "1253", "1254", "1255", "1256", "1257", "1258"
]

# Write to text file
with open("Cycled_Display_Styles.txt", "w") as file:
    for string in strings:
        current_style = next(style_cycle)
        file.write(f"STR_TEST:FIX:{current_style}:{string}\n")

print("File generated successfully!")