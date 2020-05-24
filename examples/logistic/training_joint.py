"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
from examples.logistic.common import DataOwner, LogisticRegression, ModelOwner
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

num_features = 784
alice_num_features = 392
training_set_size = 1000
test_set_size = 100
batch_size = 10
num_batches = (training_set_size // batch_size) * 10


def alice_train_data_fn():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_data = mnist.train.images[:training_set_size, alice_num_features:]
    train_labels = mnist.train.labels[:training_set_size]
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    return dataset


def alice_test_data_fn():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_data = mnist.test.images[:test_set_size, alice_num_features:]
    test_labels = mnist.test.labels[:test_set_size]
    dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    return dataset


def bob_test_data_fn():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_data = mnist.test.images[:test_set_size, alice_num_features:]
    # test_labels = mnist.test.labels[:test_set_size]
    dataset = tf.data.Dataset.from_tensor_slices(test_data)
    return dataset


def bob_train_data_fn():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_data = mnist.train.images[:training_set_size, :alice_num_features]
    # train_labels = mnist.train.labels[:training_set_size]
    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    return dataset


alice = DataOwner(
    "alice", alice_num_features, training_set_size, test_set_size, batch_size, alice_train_data_fn, alice_test_data_fn
)

x_train_0, y_train_0 = alice.preprocess_data(True, True)
x_test_0, y_test_0 = alice.preprocess_data(False, True)

bob = DataOwner(
    "bob", num_features - alice_num_features, training_set_size, test_set_size, batch_size, bob_train_data_fn,
    bob_test_data_fn
)

x_train_1 = bob.preprocess_data(True, False)

tfe.set_protocol(
    tfe.protocol.Pond(
        tfe.get_config().get_player(alice.player_name),
        tfe.get_config().get_player(bob.player_name),
    )
)
x_test_1 = bob.preprocess_data(False, False)
# x_train_1, y_train_1 = bob.provide_training_data()


x_train = tfe.concat([x_train_0, x_train_1], axis=1)
y_train = y_train_0

x_test = tfe.concat([x_test_0, x_test_1], axis=1)
y_test = y_test_0

model = LogisticRegression(num_features)
reveal_weights_op = alice.receive_weights(model.weights)

with tfe.Session() as sess:
    sess.run(
        [
            tfe.global_variables_initializer(),
            alice.initializer,
            bob.initializer,
        ],
        tag="init",
    )

    model.fit(sess, x_train, y_train, num_batches)
    # TODO(Morten)
    # each evaluation results in nodes for a forward pass being added to the graph;
    # maybe there's some way to avoid this, even if it means only if the shapes match
    model.evaluate(sess, x_test, y_test, alice)
    # model.evaluate(sess, x_test_1, y_test_1, bob)

    sess.run(reveal_weights_op, tag="reveal")
    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
