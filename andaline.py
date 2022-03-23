from random import randint  # biblioteca para gerar números aleatórios
import numpy as np
import matplotlib.pyplot as plt

class Andaline:
    def __init__(self, taxa_aprendizagem=0.0025, peso_bias=0):
        self.peso_bias = peso_bias
        self.taxa_aprendizagem = taxa_aprendizagem
        self.x = []
        self.y = 0.0
        self.w = []
        self.u = 0.0
        self.saida_esperada = []
        self.count = 0
        self.eqm_ant = float("inf")
        self.eqm_atu = 1
        self.precisao = 10**-6
        self.eqm_value_list = []
        self.epoca_value_list = []

    def separar_entradas_saida(self, vetor_dados_completos):
        self.saida_esperada = [x[(len(vetor_dados_completos))] for x in vetor_dados_completos]
        self.x = np.delete(vetor_dados_completos, [(len(vetor_dados_completos))], 1)

        # print(self.x)
        # print(self.saida_esperada)

    def plotar_grafico(self, x, y):
        print('EQM {}\nEpoca {}'.format(x, y))
        plt.plot(x, y)
        # plt.plot(x, y, 'k--')
        # plt.plot(x, y, 'go')
        plt.show()

    def inicializar_vetor_pesos(self, x):
        vetor_pesos = []
        for i in range(len(x) - 1):
            vetor_pesos.append(randint(0, 1))
        vetor_pesos.insert(0, self.peso_bias)
        return vetor_pesos

    def eqm_function(self, error):
        return (error ** 2).sum() / 2.0

    def treinar(self):
        print('\nTreinamento')
        epoca = 1
        error = 0
        file = open('treinamento.txt', 'r')
        vetor_parametros_aux = []

        for line in file:
            vetor_parametros_str = line.rstrip('\n').split(" ")
            vetor_parametros_aux.append(vetor_parametros_str)

        vetor_dados_completos = np.array(vetor_parametros_aux, dtype=np.float32)
        self.separar_entradas_saida(vetor_dados_completos)

        # self.w = self.inicializar_vetor_pesos(self.x)
        while ((self.eqm_atu - self.eqm_ant) <= self.precisao):
            self.eqm_ant = self.eqm_atu
            for i in range(len(self.x)):
                self.w = self.inicializar_vetor_pesos(self.x)
                self.count += 1
                print('\nT{}'.format(self.count))
                print('Vetor de pesos Inicial:')
                print(*self.w, sep='\t')
                eqm = 0
                for j in range(len(self.x[i]) - 1):
                    self.u += self.x[i][j + 1] * self.w[j + 1]

                self.u += self.w[0]

                error = self.saida_esperada[i] - self.u

                for j in range(len(self.x[i]) - 1):
                    self.w[j + 1] += round(self.taxa_aprendizagem * error * self.x[
                        i][j + 1], 3)

                self.w[0] += round(self.taxa_aprendizagem * error, 3)

                eqm += round(self.eqm_function(error), 3)

                self.eqm_value_list.append(eqm)
                self.epoca_value_list.append(epoca)
                self.plotar_grafico(self.epoca_value_list, self.eqm_value_list)

                print('Vetor de pesos Final:')
                print(*self.w, sep='\t')
                print('Epoca: {}'.format(epoca))
            epoca += 1
            self.eqm_atu = eqm

        self.x = []

    def testar(self):
        print('\nTeste\n')
        file = open('teste.txt', 'r')
        vetor_parametros_aux = []
        self.count = 0

        for line in file:
            vetor_parametros_str = line.rstrip('\n').split(" ")
            vetor_parametros_aux.append(vetor_parametros_str)

        self.x = np.array(vetor_parametros_aux, dtype=np.float32)

        for i in range(len(self.x)):
            self.count += 1
            print(self.count, end="\t")
            print(*self.x[i][1:], sep="\t", end="\t")

            for _ in range(5):
                for j in range(len(self.x[i]) - 1):
                    self.u += self.x[i][j + 1] * self.w[j + 1]

                self.u += self.w[0]

                self.y = -1 if self.u < 0 else 1
                print(self.y, end="\t")

            print()


andaline = Andaline()
andaline.treinar()
andaline.testar()

