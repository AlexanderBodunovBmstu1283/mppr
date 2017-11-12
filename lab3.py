import numpy as np
import sys



train=np.array([

    ([[1,1,1,1,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,1,1,1,1]],0),

    ([[0,0,1,0,0],
      [0,0,1,0,0],
      [0,0,1,0,0],
      [0,0,1,0,0],
      [0,0,1,0,0],
      [0,0,1,0,0],
      [0,0,1,0,0]],1),

    ([[1,1,1,1,1],
      [0,0,0,0,1],
      [0,0,0,0,1],
      [0,0,0,1,0],
      [0,0,1,0,0],
      [0,1,0,0,0],
      [1,1,1,1,1]],2),

    ([[1,1,1,1,1],
      [0,0,0,0,1],
      [0,0,0,0,1],
      [1,1,1,1,1],
      [0,0,0,0,1],
      [0,0,0,0,1],
      [1,1,1,1,1]],3),

    ([[1,0,0,0,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,1,1,1,1],
      [0,0,0,0,1],
      [0,0,0,0,1],
      [0,0,0,0,1]],4),

    ([[1,1,1,1,1],
      [1,0,0,0,0],
      [1,0,0,0,0],
      [1,1,1,1,1],
      [0,0,0,0,1],
      [0,0,0,0,1],
      [1,1,1,1,1]],5),

    ([[1,1,1,1,1],
      [1,0,0,0,0],
      [1,0,0,0,0],
      [1,1,1,1,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,1,1,1,1]],6),

    ([[1,1,1,1,1],
      [0,0,0,0,1],
      [0,0,0,0,1],
      [0,0,0,1,0],
      [0,0,1,0,0],
      [0,1,0,0,0],
      [1,0,0,0,0]],7),

    ([[1,1,1,1,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,1,1,1,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,1,1,1,1]],8),

    ([[1,1,1,1,1],
      [1,0,0,0,1],
      [1,0,0,0,1],
      [1,1,1,1,1],
      [0,0,0,0,1],
      [0,0,0,0,1],
      [1,1,1,1,1]],9)

]
)



import pygame
from pygame import *

# Объявляем переменные
WIN_WIDTH = 900  # Ширина создаваемого окна
WIN_HEIGHT = 640  # Высота
CELL_WIDTH=50
CELL_HEIGHT=50
DISPLAY = (WIN_WIDTH, WIN_HEIGHT)  # Группируем ширину и высоту в одну переменную
BACKGROUND_COLOR = "#ffffff"


arr=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
our=[]


def main():
    global arr
    loop = 0
    pygame.init()  # Инициация PyGame, обязательная строчка
    screen = pygame.display.set_mode(DISPLAY)  # Создаем окошко
    pygame.display.set_caption("Введите изображение, потом цифру.После ввода всех цифр - введите режим. Тренировка-белая кнопка справа. Распознавание - синяя")  # Пишем в шапку
    bg = Surface((WIN_WIDTH, WIN_HEIGHT))  # Создание видимой поверхности
    # будем использовать как фон
    bg.fill(Color(BACKGROUND_COLOR))  # Заливаем поверхность сплошным цветом

    while 1:  # Основной цикл программы
        loop+=1
        for e in pygame.event.get():  # Обрабатываем события
            if e.type == QUIT:
                our.append(arr)
                return 0
                pass
            if e.type==MOUSEBUTTONDOWN:
                if e.button == 1:  # левая кнопка мыши
                    x_ch=e.pos[1] //50-1
                    y_ch=e.pos[0]//50-1
                    if x_ch==0 and y_ch==7:
                        #print(our)
                        return 1
                    else:
                        if x_ch==1 and y_ch==7:
                            #print (our)
                            return 2
                        else:
                            if arr[x_ch][y_ch]==1:
                                arr[x_ch][y_ch]=0
                            else:
                                arr[x_ch][y_ch]=1
                    bg.fill(Color(BACKGROUND_COLOR))
            if e.type==KEYDOWN:
                    our.append([arr,e.key-48])
                    arr = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
                    bg.fill(Color(BACKGROUND_COLOR))
        screen.blit(bg, (0, 0))  # Каждую итерацию необходимо всё перерисовывать
        pygame.display.update()  # обновление и вывод всех изменений на экран
        for X in range(CELL_WIDTH, 300, CELL_WIDTH):
            for Y in range(CELL_HEIGHT, 400, CELL_HEIGHT):
                color=arr[int(Y / 50) - 1][int(X / 50) - 1]
                if color==0:
                    color_real=1
                else:
                    color_real=0
                cell=pygame.draw.rect(bg, (0,255,0), (X, Y, CELL_WIDTH, CELL_HEIGHT),color_real)
                if loop==1:
                    pass
        pygame.draw.rect(bg, (0, 255, 255), (400, 50, CELL_WIDTH, CELL_HEIGHT),2)
        pygame.draw.rect(bg, (0, 255, 255), (400, 100, CELL_WIDTH, CELL_HEIGHT), 0)




import math

#print (our)


#print(np.array_equal(train,our))

class Capture:
    global train
    def __init__(self):
        self.weights_horisontal_1=np.array([[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2]])
        self.weights_vertical_1=np.array([[0.142,0.142,0.142,0.142,0.142,0.142,0.142],[0.142,0.142,0.142,0.142,0.142,0.142,0.142],[0.142,0.142,0.142,0.142,0.142,0.142,0.142],[0.142,0.142,0.142,0.142,0.142,0.142,0.142],[0.142,0.142,0.142,0.142,0.142,0.142,0.142]])
        self.weights_horisontal_2=np.array([[0.15,0.15,0.15,0.15,0.15,0.15,0.15]])
        self.weights_vertical_2=np.array([[0.2,0.2,0.2,0.2,0.2]])
        self.weights_final=np.array([0.416,0.583])
    def sigmoid_mapper(self,arr):
        result=arr#1 / (1 + np.exp(-2*arr))
        return result
    def check(self,num,arr):
        hor1 = self.check_horisontal_1(num,arr)
        self.hor2 = self.check_horisontal_2(hor1)
        #print(self.hor2)
        ver1 = self.check_vertical_1(num,arr)
        self.ver2 = self.check_vertical_2(ver1)
        #print(self.ver2)
        final = self.check_final(self.hor2, self.ver2)
        #print(final)
        return final
    def check_horisontal_1(self,a,arr):
        result=[]
        equal=np.array(train[a][0])==np.array(arr)
        #print(equal)
        for i in range(len(equal)):
            row=self.check_horisontal_1_layer(equal[i],self.weights_horisontal_1[i])
            #print(row)
            result.append(row)
        #print("\n")
        result=self.sigmoid_mapper(np.array(result))
        return result
    def check_horisontal_1_layer(self,row,row_weight):
        return np.dot(row,row_weight)
    def check_horisontal_2(self,arr_row):
        for i in arr_row:
            return self.sigmoid_mapper(np.dot(arr_row,self.weights_horisontal_2[0]))
    def check_vertical_1(self,a,arr):
        result=[]
        equal = (np.array(train[a][0]) == np.array(arr)).transpose()
        #print('transpose: ',equal)
        for i in range(len(equal)):
            col=self.check_vertical_1_layer(equal[i],self.weights_vertical_1[i])
            #print(col)
            result.append(col)
        #print("\n")
        return self.sigmoid_mapper(np.array(result))
    def check_vertical_1_layer(self,col,col_weight):
        return np.dot(col,col_weight)
    def check_vertical_2(self,arr_row):
        for i in arr_row:
            return self.sigmoid_mapper(np.dot(arr_row,self.weights_vertical_2[0]))
    def check_final(self,hor,vert):
        self.final=np.dot([hor,vert],self.weights_final)
        return self.final

class Train(Capture):
    def __init__(self):
        super().__init__()
        self.learning_rate=0.05
    def train(self,num,arr,expected):
        actual_predict=self.check(num,arr)
        actual_predict=math.atan(actual_predict)/1.57#1 / (1 + np.exp(-actual_predict))
        self.expected=expected
        self.num=num
        expected=self.form_expected()
        #print("Actual predict:",actual_predict)
        #print([self.hor2,self.ver2])
        error_layer_2 = np.array([actual_predict-expected])
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        dx=((np.dot(weights_delta_layer_2, np.reshape([self.hor2,self.ver2],(1,2)))) * self.learning_rate)[0]
        self.weights_final[0] += dx
        self.weights_final[1] -= dx
        #print(np.dot(weights_delta_layer_2, self.final))
        #print(self.weights_final)
        #print(expected)
        return actual_predict
    def form_expected(self):
        return self.num==self.expected

class Recognize(Train):
    def __init__(self):
        super().__init__()
        self.is_recognized=[]
    def recognize(self, arr):
        for i in arr:
            predict=[]
            for num in range(0,10):
                actual_predict = self.check(num, i[0])
                actual_predict = math.atan(actual_predict)  # 1 / (1 + np.exp(-actual_predict))
                self.expected = i[0][1]
                self.num = num
                predict.append(actual_predict)
            max = predict[0]
            pos = 0

            for k in range(len(predict)):
                if predict[k] > max:
                    max = predict[k]
                    pos = k
            self.is_recognized.append(i[1] == pos)
            print("Степень подобия введенной цифры цифрам 0-9:")
            for i in predict:
                print(i)
            print("Степень подобия введенной цифры цифрам 0-9 *******")
        return self.is_recognized
    def form_expected(self):
        return self.num==self.expected





def recognize():
    a=Recognize()
    print("Результаты распознавания: ",a.recognize(our1))

def our_train():
    b=Recognize()
    res=[]
    for i in range(0,10):
        res.append(b.train(i,our1[0][0],our1[0][1]))
        #res.append(9)
    print("Результаты распознавания после обучения: ",b.recognize(our1))

if __name__ == "__main__":
    mod=main()
    our1 = np.array(our)
    print("Введенные данные :  ",our1)
    if (mod==1):
        our_train()
    else:
        if (mod==2):
            recognize()


