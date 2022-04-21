import re
import time

start_time = time.time()
regex = r"([\w ]+).*(\d{4})"
f = open("movies.txt", "r")
test_str = f.read()
matches = re.finditer(regex, test_str)

for matchNum, match in enumerate(matches, start=1):
    # res = match.group(1), match.group(2), match.group(3)
    print(match.group(1), ",", match.group(2))

end_time = time.time()
result = end_time - start_time

print("completed in: {:.2f} seconds".format(result))