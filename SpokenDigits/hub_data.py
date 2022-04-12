import hub
import matplotlib.pyplot as plt

ds = hub.load("hub://activeloop/spoken_mnist")


# check out the first spectrogram, it's label, and who spoke it!
plt.imshow(ds.spectrograms[0].numpy())
plt.title(f"{ds.speakers[0].data()} spoke {ds.labels[0].numpy()}")
plt.show()

