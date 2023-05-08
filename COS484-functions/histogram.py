import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from the file
data = pd.read_csv('COS484-data/validation.masked.tsv', sep='\t', header=None, names=['sentence', 'truthWord'])

# Calculate the distribution of truthWords
truthWord_counts = data['truthWord'].value_counts()

# Specify the order of the x-axis labels
order = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "no", "zero"]

# Reorder the truthWord_counts based on the specified order
ordered_counts = truthWord_counts.loc[order]

# Generate a pleasant-looking color palette
colors = sns.color_palette("husl", len(ordered_counts))

# Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(ordered_counts.index, ordered_counts.values, color=colors)
plt.xlabel('Truth Word')
plt.ylabel('# of Examples')
plt.title('Distribution of Truth Words in the Test Data (validation.masked.tsv)')

# Save the plot as an image file
plt.savefig('truthWord_TestSet.png')

# Show the plot
plt.show()