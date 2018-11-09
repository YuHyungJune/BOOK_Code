import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime as dt
import numpy as np
import pickle as pkl
from functools import partial

EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.001
ALPHA = 0.01

# 생성모델을 만드는 함수 
def generator(randomData, alpha, reuse=False):
    with tf.variable_scope('GAN/generator', reuse=reuse):
        # hidden layer
        hl = tf.layers.dense(randomData, 256, activation=partial(tf.nn.leaky_relu, alpha=alpha))
        # output layer
        ol = tf.layers.dense(hl, 784, activation=None)
        # 활성화함수 tanh
        img = tf.tanh(ol)
        
        return ol

# 식별모델을 만드는 함수 
def discriminator(img, alpha, reuse=False):
        with tf.variable_scope('GAN/discriminator', reuse=reuse):
            # hidden layer
            hl = tf.layers.dense(img, 128, activation=partial(tf.nn.leaky_relu, alpha=alpha))
            # output lyaer
            D_logits = tf.layers.dense(hl, 1, activation=None)
            # 활성화함수 sigmoid
            D = tf.nn.sigmoid(D_logits)
        return D, D_logits
    
if __name__ == '__main__':
    tstamp_s = dt.datetime.now().strftime("%H:%M:%S")
    mnist = input_data.read_data_sets('MNIST_DataSet')
    print("Start....", tstamp_s)
    # 원본이미지 데이터(784차원)을 배치사이즈만큼 보관하는 홀더 준비
    ph_realData = tf.placeholder(tf.float32, (BATCH_SIZE, 784))
    # 100차원의 균일 난수를 보관하는 홀더 준비
    # 확보한 사이즈는 학습시는 배치사이즈 100건, 각 에포크에서 이미지 생성시는 25건, 
    # 동적으로 변하기 때문에 None을 지정하고,
    # 실행시 사이즈를 결정하도록 함
    ph_randomData = tf.placeholder(tf.float32, (None, 100))
    
    # 균일난수를 부여하여 이미지 생성 
    gimage = generator(ph_randomData, ALPHA)
    # 원본이미지를 부여하여 판별 결과를 취득
    real_D, real_D_logits = discriminator(ph_realData, ALPHA)
    # 생성이미지를 부여하여 판별 결과를 취득
    fake_D, fake_D_logits = discriminator(gimage, ALPHA, reuse=True)
    
    # 손실 함수 실장
    # 원본이미지(Label=1)과의 오차를 크로스엔트로피(crossentropy)의 평균으로 취득 
    d_real_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits, labels=tf.ones_like(real_D))
    
    loss_real = tf.reduce_mean(d_real_xentropy)
    
    # 생성이미지(Label=0)과의 오차를 크로스엔트로피(crossentropy)의 평균으로 취득 
    d_fake_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits, labels=tf.zeros_like(fake_D))
    
    loss_fake = tf.reduce_mean(d_fake_xentropy)
    
    # Discriminator의 오차와 원본이미지, 생성이미지의 오차를 더함
    d_loss = loss_real + loss_fake
    
    # Generator의 오차를 취득 
    g_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits, labels=tf.ones_like(fake_D))
    g_loss = tf.reduce_mean(g_xentropy)
    
    # 학습에의한 최적화를 하는 파라미터(Weight, Bias)를 tf.trainable_variables로부터 일괄 취득
    # 취득할 때에 Discriminator용(d_training_parameter), Generator용(g_training_parameter)으로 
    # 나누어 각각 네트워크를 최적화할 필요가 있기 때문에 네트워크 정의시 지정한 Scope의 이름을 지정해서 나눔
    # discriminator의 최적화를 하는 학습 파라미터를 취득(일단, trainVar에 나누어서 저장)
    d_training_parameter = [trainVar for trainVar in tf.trainable_variables() if 'GAN/discriminator/' in trainVar.name]
    
    # genrator의 최적화을 하는 학습 파라미터를 취득(일단, trainVar에 나눠서 저장)
    g_training_parameter = [trainVar for trainVar in tf.trainable_variables() if 'GAN/generator/' in trainVar.name]
    
    # AdamOptimizer로 학습 파라미터의 최적화를 실행
    # 일괄 취득한 Discriminator의 파라미터 갱신 
    d_optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_training_parameter)
    
    # 일괄 취득한 Generator의 파라미터 갱신
    g_optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_training_parameter)
    
    batch = mnist.train.next_batch(BATCH_SIZE)
    
    # 진행 준간의 내용을 저장 할 변수 정의 
    save_gimage = []
    save_loss = []
    
    # (5)학습처리의 실장
    with tf.Session() as sess:
        # 변수 초기화 
        sess.run(tf.global_variables_initializer())
        
        # EPOCH수 만큼 반복
        for e in range(EPOCHS):
            # 배치사이즈 100
            for i in range(mnist.train.num_examples//BATCH_SIZE):
                batch = mnist.train.next_batch(BATCH_SIZE)
                batch_images = batch[0].reshape((BATCH_SIZE, 784))
                
                # generator에서 활성화함수 tanh를 사용하기 때문에 range를 맞춤 
                batch_images = batch_images * 2 - 1
                # generator에 전달하는 균일분포 랜덤 노이즈를 생성
                # 값은 -1 ~ 1까지, 사이즈는 batch_size * 100
                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
                
                # 최적화 계산, 파라미터 갱신을 함
                # Discriminator의 최적화를 사용하는 데이터군을 feed_dict로 제공 
                sess.run(d_optimize, feed_dict={ph_realData:batch_images, ph_randomData:batch_z})
                # Generator의 최적화와 최적화를 사용하는 데이터군을 feed_dict로 제공 
                sess.run(g_optimize, feed_dict={ph_randomData:batch_z})
            
            # training의 loss를 기록
            train_loss_d = sess.run(d_loss, {ph_randomData:batch_z, ph_realData:batch_images})
            # eval은 generator의 loss(g_loss)를 출력하는 명령
            train_loss_g = g_loss.eval({ph_randomData:batch_z})
            
            # 학습 과정을 표시
            print("{0} Epoch={1}/{2}, DLoss={3:.4F}, GLoss={4:.4F}".format(
                dt.datetime.now().strftime("%H:%M:%S"), e+1, EPOCHS, train_loss_d, train_loss_g))
            
            # loss을 저장하기 위해서 리스트에 추가 
            # train_loss_d, train_logg_g를 SET으로 리스트에 추가하고 후에 가시화 가능하도록 함 
            save_loss.append((train_loss_d, train_loss_g))
            
            # 학습 도중의 생성 모델로 이미지를 생성하고 보존
            # 균일 난수 데이터를 25개 생성하고, 그 데이터를 사용하여 이미지를 생성하고 보존
            randomData = np.random.uniform(-1,1, size=(25,100))
            
            # gen_samples에 현시점의 모델로 만든 데이터를 읽도록 유지
            # 노이즈, 사이트, 유닛수(128), reuse는 상태유지, 데이터는 randomData로서 feed_dict에 지정 
            gen_samples = sess.run(generator(ph_randomData, ALPHA, True), feed_dict={ph_randomData:randomData})
            # 생성 이미지를 보존 
            save_gimage.append(gen_samples)
            
    # pkl 형식으로 생성이미지를 보존
    with open('save_gimage.pkl', 'wb') as f:
        pkl.dump(save_gimage, f)
    
    # 각 EPHOCH에서 얻은 손실함수의 값을 보존
    with open('save_loss.pkl', 'wb') as f:
        pkl.dump(save_loss, f)
    
    # 처리 종료 시각을 취득
    tstamp_e = dt.datetime.now().strftime("%H:%M:%S")
    
    time1 = dt.datetime.strptime(tstamp_s, "%H:%M:%S")
    time2 = dt.datetime.strptime(tstamp_e, "%H:%M:%S")
    
    print("시작:{0}, 종료:{1}, 처리시간:{2}".format(tstamp_s, tstamp_e, (time2-time1)))
