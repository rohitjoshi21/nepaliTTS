import finalmodel as fm


text = input("Enter you text: ")
audio, sr = fm.end_to_end_infer(text, fm.pronounciation_dictionary,fm.show_graphs)

print(audio)

print(sr)