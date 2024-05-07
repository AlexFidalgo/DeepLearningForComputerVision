

f = plt.figure()

for i in range(20):
    f.add_subplot(4,5, i+1)
    plt.imshow( [i-ésima imagem], cmap='gray')
    plt.axis("off")
    plt.text(0,-3, [i-ésima classificação verdadeira], color='b')
    plt.text(0,2,[i-ésima classificação do algoritmo], color='r')
plt.savefig("nome_da_imagem.png")
plt.show()