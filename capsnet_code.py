
"""
01.04.2020
Capsule Neural Networks:  introduction and its implementation in classification data base MNIST

Данный код представляет собой капсульную нейронную сеть,
задача которой -  классицикация базы данных MNIST
"""


#Импортирование требуемых библиотек
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import pandas as pd


"""
Reproducibility Images
"""
#Воспроизведение изучаемых изображений
tf.compat.v1.reset_default_graph()
np.random.seed(42)
tf.compat.v1.set_random_seed(42)


"""
Load up the data base MNIST
"""
#Загрузка базы данных MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#Визуализация произвольной части исследуемой базы данных
n_samples = 5
"""

plt.figure(figsize=(n_samples * 3, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_image = train_images[index].reshape(28, 28)
    plt.imshow(sample_image, cmap="binary")
    plt.axis("off")
#plt.show()

#Визуализация меток, соответствующих своему изображению
#print(train_labels[:n_samples])
"""

"""
Input Images
"""
#Предаствление набора данных(набора изображений) в виде тензора или же 3-х мерного массива
X = tf.compat.v1.placeholder(shape=[n_samples, 28, 28, 1], dtype=tf.float32, name="X")
#print(X)


"""
Primary Capsules
"""
#Инициализация параметров для сверточных слоев
caps1_n_maps = 32 #количество частей, на которые делится  полученный набор данных
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 первичные капсулы
caps1_n_dims = 8 #размерность первичных капсул


# tf.keras.layers.Conv2D - If using Keras pass *_constraint arguments to layers ???

#Первый сверточный слой
#кол-во фильтров  - 256
#размер фильтра - 9х9
#шаг фильтра - 1
conv1 = tf.keras.layers.Conv2D(
    filters=256, kernel_size=9, strides=(1, 1), padding='valid',
    dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=tf.keras.regularizers.l2(0.05), name="conv1")
out_conv1 = conv1(X)
#print(out_conv1)


#Второй сверточный слой
#кол-во фильтров  - 256
#размер фильтра - 9х9х256
#шаг фильтра - 2
conv2 = tf.keras.layers.Conv2D(
    filters=256, kernel_size=9, strides=(2, 2), padding='valid',
    dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=tf.keras.regularizers.l2(0.05), name="conv2")
out_conv2 = conv2(out_conv1)
#print(out_conv2)


#Разделение полученного стека данных на  1152 вектора  размерностью 8
caps1_raw = tf.reshape(out_conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
#print(caps1_raw)


#Теперь каждый полученный  вектор  необходимо нормировать, так как его длина может превышать 1
#Для этого воспользуемся новой нелинейной  функцией squash(s)
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.math.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.math.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
caps1_output = squash(caps1_raw, name="caps1_output")
#print(caps1_output)


"""
Digit Capsules
"""
#Инициализация параметров для слоя с цифровыми капсулами
caps2_n_caps = 10 #количество капсул соответствует количеству классов
caps2_n_dims = 16 #размерность цифровых капсул
init_sigma = 0.1 #Область  отклонения случайных значений матриц W


#Инициализация обучаемых матриц преобразования для каждой первичной капсулы предсказаний
W_init = tf.random.normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")
#print(W)


batch_size = tf.shape(X)[0] #количество наборов матриц W соответствует количеству обрабатываемых  изображений
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled") #создание копий наборов матриц W, количество которых соответствует количеству изображений
#print(W_tiled)
#ВНИМАНИЕ! Индексирование размеронстей идет справа налево!(индексирование как  обычно начинается с 0)
caps1_output_expanded = tf.compat.v1.expand_dims(caps1_output, -1, name="caps1_output_expanded") #создание размерности под  индексом -1(размерность  будет равна 1 и будет стоять на 1 месте справа в скобках)
#print(caps1_output_expanded)
caps1_output_tile = tf.compat.v1.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile") #создание размерности под  индексом 2(размерность  будет равна 1 и будет стоять на 2 месте справа в скобках)
#print(caps1_output_tile)
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled") #создание копий набора первичных капсул, количество которых соответствует количеству классов
#print(caps1_output_tiled)
caps2_predicted = tf.linalg.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted") #капсулы, полученные в результате скалярного перемножения матриц W и первичных векторов
print(caps2_predicted)


"""
Routing by Agreement
"""

#Инициализируем веса предсказаний, присваивая им начальные значения равные 0
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")
#Применение функции softmax
routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")
#Перемножение весов и векторов предсказаний
weighted_predictions = tf.math.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
#Вычисляется сумма произведений векторов и весов
weighted_sum = tf.math.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
#Нормирование полученного вектора
caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")
#print(caps2_output_round_1)


caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1], name="caps2_output_round_1_tiled")
#Скалярное перемножение капсул текущего и предыдущего уровней
agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled, transpose_a=True, name="agreement")
#Обновление весов для следующего раунда
raw_weights_round_2 = tf.math.add(raw_weights, agreement, name="raw_weights_round_2")


#2-ой раунд аналогичен первому
routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2, axis=2, name="routing_weights_round_2")
weighted_predictions_round_2 = tf.math.multiply(routing_weights_round_2, caps2_predicted, name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.math.reduce_sum(weighted_predictions_round_2, axis=1, keepdims=True, name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2, axis=-2, name="caps2_output_round_2")
caps2_output = caps2_output_round_2
print(caps2_output)


#Очередная функция squash для нормирования векторов без изменения направления
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.math.reduce_sum(tf.math.square(s), axis=axis, keepdims=keep_dims)
        return tf.math.sqrt(squared_norm + epsilon)


y_proba = safe_norm(caps2_output, axis=-2, name="y_proba") #Нормализация всех векторов
y_proba_argmax = tf.math.argmax(y_proba, axis=2, name="y_proba_argmax") #Находим индекс вектора с наибольшим значением
y_pred = tf.compat.v1.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred") #Избавляемся от последних 2-х измерений(отсчет идет справа налево)
y = tf.compat.v1.placeholder(shape=[None], dtype=tf.int64, name="y") #Тензор, содержащий  в себе классы цифр 0..9


#Значения, требуемые для margin loss
m_plus = 0.9 # m^+
m_minus = 0.1 #  m^-
lambda_ = 0.5 # lambda


#Значение T_k  для каждого экземпляра и класса
T = tf.one_hot(y, depth=caps2_n_caps, name="T")


#with tf.compat.v1.Session():
#print(T.eval(feed_dict={y: np.array([0, 1, 2, 3, 4, 5])}))


caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm") #Вычисляем норму векторов предсказаний
present_error_raw = tf.math.square(tf.math.maximum(0., m_plus - caps2_output_norm), name="present_error_raw") #Вычисляем max(0,m^+ - ||v_k||)^2
present_error = tf.reshape(present_error_raw, shape=(-1, 10), name="present_error") #Перегруппируем результат
absent_error_raw = tf.math.square(tf.math.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw") #Вычисляем max(0,||v_k|| - m^-)^2
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10), name="absent_error") #Перегруппируем результат
L = tf.math.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L") #Считаем потерю данных для каждой  цифры
margin_loss = tf.math.reduce_mean(tf.math.reduce_sum(L, axis=1), name="margin_loss") #Вычисляем среднее значение от  суммы всех  потерь данных для  каждой  цифры


#Реконструкция  исходного  изображения
mask_with_labels = tf.compat.v1.placeholder_with_default(False, shape=(), name="mask_with_labels") #Тензор, в котором содержаться скрытые  векторы, не требуемые на  выходе
reconstruction_targets = tf.compat.v1.cond(mask_with_labels, lambda: y, lambda: y_pred,  name="reconstruction_targets") #Определяем объект для реконструкции, используя метки y - True, y_pred - False
reconstruction_mask = tf.one_hot(reconstruction_targets, depth=caps2_n_caps, name="reconstruction_mask") #Маска для реконструкции(равна 1 для требуемого  объекта, 0  - для иного)
reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1], name="reconstruction_mask_reshaped") #Изменение размерности для возможности перемножения
caps2_output_masked = tf.math.multiply(caps2_output, reconstruction_mask_reshaped, name="caps2_output_masked") #Перепножение масок с выходными капсулами
decoder_input = tf.reshape(caps2_output_masked, [-1, caps2_n_caps * caps2_n_dims], name="decoder_input") #Снова изменяем размерность тензора
print(decoder_input)


#Decoder
n_hidden1 = 512 #Количество нейронов в первом скрытом слое
n_hidden2 = 1024 #Количество нейронов во втором скрытом слое
n_output = 28 * 28 #Количество нейронов в выходном слое(изображение 28х28)


#Построение декодера
hidden1 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros')(decoder_input) #Первый скрытый слой
hidden2 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros')(hidden1) #Второй скрытый слой
decoder_output = tf.keras.layers.Dense(n_output, activation=tf.nn.sigmoid, kernel_initializer='glorot_uniform', bias_initializer='zeros')(hidden2) #Выходной слой


#Вычисление  потер данных при реконструкции
X_flat = tf.reshape(X, [-1, n_output], name="X_flat") #Изменение размерности тензора Х
squared_difference = tf.math.square(X_flat - decoder_output, name="squared_difference") #Вычисление квадрата разности между входным и воссоздаваемым изображениями
reconstruction_loss = tf.math.reduce_mean(squared_difference, name="reconstruction_loss") #Вычисление среднего значения результата


#Вычисление общей потери данных как сумма 2-х  значений
alpha = 0.0005 #Коэффициент масштабирования
loss = tf.math.add(margin_loss, alpha * reconstruction_loss, name="loss")
#print(loss)


#Вычисляем количество экзэмплярогв, которые были правильно классифицированы
correct = tf.math.equal(y, y_pred, name="correct") #Определяем, какая метка активна
accuracy = tf.math.reduce_mean(tf.cast(correct, tf.float32), name="accuracy") #Вычисляем среднее значение всех требуемых меток


#Операции для тренировки нейросети
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='optimizer') #Алгоритм оптимизации тренировки нейросети
training_op = optimizer.minimize(loss, name="training_op") #Оптимизация минимизации потери данных                       !!! - Instructions for updating: Use tf.where in 2.0, which has the same broadcast rule as np.where ???


init = tf.compat.v1.global_variables_initializer() # ???
saver = tf.compat.v1.train.Saver()

#Стандартная тренировка нейросети
n_epochs = 10 #Количество эпох
batch_size = n_samples #Количество изображений
restore_checkpoint = True


n_iterations_per_epoch = 5000 // batch_size #Количество итераций за эпоху
n_iterations_validation = 1000 // batch_size #Количество итераций подтверждения
best_loss_val = np.infty #Значение бесконечности
checkpoint_path = "./CNN"


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

with tf.compat.v1.Session() as sess:
    if restore_checkpoint and tf.compat.v1.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            _, loss_train = sess.run([training_op, loss], feed_dict={X: X_batch.reshape([-1, 28, 28, 1]), y: y_batch, mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(iteration, n_iterations_per_epoch, iteration * 100 / n_iterations_per_epoch, loss_train), end="")

        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = mnist.validation.next_batch(batch_size)
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_batch.reshape([-1, 28, 28, 1]), y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_validation, iteration * 100 / n_iterations_validation), end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(epoch + 1, acc_val * 100, loss_val, " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val


n_iterations_test = 1000 // batch_size

with tf.compat.v1.Session() as sess:
        saver.restore(sess, checkpoint_path)
        loss_tests = []
        acc_tests = []
        for iteration in range(1, n_iterations_test + 1):
            X_batch, y_batch = mnist.test.next_batch(batch_size)
            loss_test, acc_test = sess.run([loss, accuracy], feed_dict={X: X_batch.reshape([-1, 28, 28, 1]), y: y_batch})
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_test, iteration * 100 / n_iterations_test), end=" " * 10)
        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(acc_test * 100, loss_test))


sample_images = mnist.test.images[:n_samples].reshape([-1, 28, 28, 1])

with tf.compat.v1.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run([caps2_output, decoder_output, y_pred], feed_dict={X: sample_images, y: np.array([], dtype=np.int64)})

sample_images = sample_images.reshape(-1, 28, 28)
reconstructions = decoder_output_value.reshape([-1, 28, 28])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(mnist.test.labels[index]))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")

plt.show()