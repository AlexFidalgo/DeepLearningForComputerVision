f = plt.figure()
for i in range(QEX.shape[0]):
  f.add_subplot(1,QEX.shape[0],i+1)
  plt.imshow( QEX[i], cmap="gray")
  plt.axis("off");
  plt.text(0,-3,QEY[i],color="b")
  plt.text(0, 2,QEP[i],color="r")
plt.savefig("QE.png")
plt.show()