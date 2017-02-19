import numpy as np
import tensorflow as tf
from PIL import Image

from src.graph import Graph
from src.environment import GymEnvironment


def test_graph():
    env = GymEnvironment('MsPacman-v0')
    graph = Graph(actions=10)
    env.reset()
    screenshots = []
    g = graph.get_graph()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for x in range(1, 7):
            env.render()
            observation, reward, done, info = env.step(0)
            screenshots.append(observation)
            if x % 4 == 0:
                concat_image = np.concatenate((screenshots[0], screenshots[1], screenshots[2], screenshots[3]), axis=1)
                im = Image.fromarray(concat_image).convert('LA')
                # im.show()
                grayscale_im = np.array(im)

                graph.run_graph(sess, grayscale_im.reshape([-1, 25600]))


if __name__ == '__main__':
    test_graph()