#Author: Joyjit Chatterjee
#Acknowledgments: Tensorflow and Deep Learning With Tensorflow Course by Cognitive Class

# https://www.tensorflow.org/versions/r0.9/get_started/index.html
# https://courses.cognitiveclass.ai/courses/course-v1:CognitiveClass+ML0120ENv2+2018/courseware/407a9f86565c44189740699636b4fb85/d82ba5edac4f40efa334fff96b944b34/
import tensorflow as tf

graph_1 = tf.Graph()
with graph_1.as_default():
    var_a = tf.constant([1], name = 'var_a_scalar')
    var_b = tf.constant([1,2,3], name = 'var_b_vector')
    var_c = tf.constant([[1,2,3], [4,5,6]], name = 'var_c_matrix')
    var_d = tf.constant([[[1,2,3], [4,5,6],[7,8,0]],[[10,11,12],[13,15,16],[66,44,12]],
                        [[9,1,2],[5,2,6],[6,7,8]]], name = 'var_d_tensor')
    
    
with tf.Session(graph = graph_1) as sess:
    res = sess.run(var_a)
    print("Value of var_a is %s \n" %res)
    res = sess.run(var_b)
    print("Value of var_b is %s \n" %res)
    res = sess.run(var_c)
    print("Value of var_c is %s \n" % res)
    res = sess.run(var_d)
    print("Value of var_d is %s \n" % res)

print("Shape of var_d (tensor) is: ", tf.shape(var_d))

#performing addition operation on the tensors in a new graph

graph_2 = tf.Graph()
with graph_2.as_default():
    matrix_a = tf.constant([[1,2,3],[4,5,6]], name = 'matrix_a')
    matrix_b = tf.constant([[7,8,9],[10,11,12]], name = 'matrix_b')
    
    #defining addition operation and creating a new tensor out of it
    matrix_total = tf.add(matrix_a, matrix_b)
 # could also use matrix_total = matrix_a + matrix_b

with tf.Session(graph = graph_2) as sess:
    final_result = sess.run(matrix_total)
    print(f"{matrix_a}" + f"+{matrix_b}" + f"={final_result}")
    

# doing Hadamard Product (element wise-multiplication of 2 matrices)
graph_3 = tf.Graph()
with graph_3.as_default():
    mat_a = tf.constant([[1,2,3],[1,1,1]])
    mat_b = tf.constant([[0,1,2], [1,0,1]])
    
    #perform element wise multiplication
    mat_hadamard = mat_a * mat_b

with tf.Session(graph = graph_3) as sess:
    res_hadamard = sess.run(mat_hadamard)
    print("Hadamard product = %s" %res_hadamard)
    
# now perform normal matrix multiplication using in-built tensorflow function
graph_4 = tf.Graph()
with graph_4.as_default():
    newmat1 = tf.constant([[1,1],[6,7]])
    newmat2 = tf.constant([[4,5], [1,7]])
    
    #perform matrix multiplication using matmult()
    final_mul = tf.matmul(newmat1,newmat2)

with tf.Session(graph = graph_4) as sess:
    res_mul = sess.run(final_mul)
    print("Final matrix multiplied product is %s" %res_mul)
    
    
#Using variables in tensorflow (variables can be updated/value changed unlike tensors)
graph_5 = tf.Graph()

with graph_5.as_default():
    var_new = tf.Variable(0)
    #initialise the global variables to use within Session later
    init_ops = tf.global_variables_initializer()

    update_value = tf.assign(var_new, var_new + 1) #update var_new and increment it by 1 for each successive 
#run
with tf.Session(graph = graph_5) as sess:
    sess.run(init_ops)
    print(sess.run(var_new))
    for _ in range(5):
        print(sess.run(update_value))
        
#Using placeholders (holders/holes) in our program
a = tf.placeholder(tf.float32) #32 bit floating point based placeholder

c = a * 5

with tf.Session() as sess:
    final_res = sess.run(c, feed_dict ={a:[[1,2,3],[4,5,6],[7,8,9]]}) # we can feed 
    #in any generic tensor in here
    print("Final result is %s" %final_res)
    
#Tensorflow operations
#Eg. tf.nn.sigmoid(), tf.add(), tf.subtract(), tf.matmul() etc......

graph_final = tf.Graph()
with graph_final.as_default():
    var_a = tf.constant([4], name = 'var_a')
    var_b = tf.constant([7], name = 'var_b')
    
    add_opn = tf.add(var_a, var_b)
    sub_opn = tf.subtract(var_b, var_a)

with tf.Session(graph = graph_final) as sess:
    final_add_res = sess.run(add_opn)
    print("Final add result is %s" %final_add_res)
    final_sub_res = sess.run(sub_opn)
    print("Final subtraction result is %s" %final_sub_res)
    

