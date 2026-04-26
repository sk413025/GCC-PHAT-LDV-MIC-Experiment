import pdfplumber

with pdfplumber.open("main.pdf") as pdf:
    # Check page 4
    if len(pdf.pages) >= 4:
        p4 = pdf.pages[3]
        text4 = p4.extract_text()
        words4 = p4.extract_words()
        if words4:
            bottom_word = max(words4, key=lambda w: w["bottom"])
            print(f"Page 4 lowest text is at Y={bottom_word['bottom']} (Page height: {p4.height})")
    
    # Check page 5
    if len(pdf.pages) >= 5:
        p5 = pdf.pages[4]
        text5 = p5.extract_text()
        words5 = p5.extract_words()
        if words5:
            # find where references start
            top_word = min(words5, key=lambda w: w["top"])
            print(f"Page 5 highest text is at Y={top_word['top']}")
            if "References" in text5:
                print("Page 5 contains References.")
