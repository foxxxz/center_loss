import tensorflow as tf

def center_loss(embedding_layer, centroids, y, params):
    loss_list = []
    y = tf.cast(y, tf.int32)
    for i in params['labels']:
        positive_sample_mask = tf.equal(y, i)
        positive_embedding = tf.boolean_mask(embedding_layer, positive_sample_mask)
        loss_list.append(0.5 * tf.reduce_sum(tf.square(centroids[i] - positive_embedding)))
    center_loss = params['lambda'] * tf.reduce_sum(tf.stack(loss_list, axis=0))
    return center_loss