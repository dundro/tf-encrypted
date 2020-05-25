import tf_encrypted as tfe
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

remote_config = tfe.RemoteConfig.load("config.json")
tfe.set_config(remote_config)

tfe.set_protocol(tfe.protocol.Pond())
players = remote_config.players
server0 = remote_config.server(players[0].name)

tfe.set_protocol(tfe.protocol.Pond(
    tfe.get_config().get_player("alice"),
    tfe.get_config().get_player("bob")
))

i = np.random.randint(mnist.test.images.shape[1])
print(i, mnist.test.labels[i])

first_image_half = mnist.test.images[i][:392]
first_image_half = np.array(first_image_half, dtype='float')
pixels = first_image_half.reshape((14, 28))

w = tfe.define_private_variable(
    pixels
)
w_masked = tfe.mask(w)


def printMasked(p):
    f = p.unwrapped[0]
    return tf.print(f.value)

with tfe.Session() as sess:
    print(pixels)
    sess.run(printMasked(w_masked))

# p_masked = tfe.mask(tf.convert_to_tensor(pixels))
# tfe.set_protocol(
#     tfe.protocol.Pond(
#         tfe.get_config().get_player(alice.player_name),
#         tfe.get_config().get_player(bob.player_name),
#     )
# )
