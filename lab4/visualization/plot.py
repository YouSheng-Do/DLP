import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loss.csv")

df = df.drop([0, 1])

plt.plot(df["Epoch"], df["Training loss"], label='training loss')
plt.plot(df["Epoch"], df["Validation loss"], label='validation loss')

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig('revised_loss_curve.png')