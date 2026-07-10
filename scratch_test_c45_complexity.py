import os

rules_path = os.path.join("outputs", "rules", "rules.py")

with open(rules_path, 'r', encoding='UTF-8') as f:
    lines = f.readlines()

rules_count = 0
total_depth = 0
depths = []

for line in lines:
    stripped = line.strip()
    if stripped.startswith("return "):
        val = stripped.split("return ")[1].replace("'", "").replace('"', '')
        if val in ['Positive', 'Negative']:
            rules_count += 1
            # Calculate indentation
            leading_spaces = len(line) - len(line.lstrip(' '))
            # In rules.py:
            # def findDecision(obj) is at 0 spaces
            # if obj[0]>6: is at 3 spaces
            # return is at leading_spaces
            # The number of conditional checks is leading_spaces // 3 - 1
            depth = (leading_spaces // 3) - 1
            depths.append(depth)
            print(f"Rule {rules_count}: returns {val} at depth {depth} (spaces: {leading_spaces})")

print(f"\nTotal rules: {rules_count}")
print(f"Depths: {depths}")
print(f"Mean depth: {sum(depths) / len(depths) if depths else 0:.2f}")
