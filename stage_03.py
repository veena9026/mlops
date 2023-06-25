
with open("artifacts.txt","r") as f:
    text=f.read()

print(text)

with open("artifacts.txt","w") as f:
    text=f.write(text+"i have add one list")

print(text)
print("i have write in stage 03")