with open("C:\\Users\\amb3942\\Desktop\\walkie-tracker\\output1.csv", "w") as file:
    for i in range(1, 826):
        file.write(f"STR_TEST:FIX:0038:{i:04d}\n")