# A simple Early stopping implementation in Tensorflow

'''Early stoppig
This function will take iteration number and early stopping starting point
evaluate validation error and if the validation error keep increasing continuously k times then exit the training,
also keep track of the best model, save it in path1/improved_model_..., and also save the corresponding best prediction'''

update_num = 0
start_erlystop = 6
earlystop_num = 0
earlystop_flag = 0
opt_val_loss_prev=100
best_val_loss = 100

def early_stopping(i, update_num, earlystop_flag, earlystop_num, best_val_loss, opt_val_loss_prev, path1):
    every_k = 10 ### each every 10 update we will check the validation error
    exit_k = 5 ## exit training if the validation error increase continously 5 times
##################################################################    
    if i > start_erlystop and update_num % every_k== 0:
        opt_val_loss= sess.run(loss_f, feed_dict={x: val_x, y_: val_y,'phase:0': 0})  
        ##keep the best model##################
        if opt_val_loss < best_val_loss:
            best_val_loss = opt_val_loss
            best_update_num = update_num
            best_test_error,  best_prediction = sess.run([loss_f, y], feed_dict={x: test_x, y_: test_y, 'phase:0': 0})
            saver.save(sess, path1 +'/improved_model', global_step = best_update_num) 
        
        ### early stopping ######################################################################
        if opt_val_loss > opt_val_loss_prev:
            earlystop_num = earlystop_num +1
        else: 
            earlystop_num = 0
                
        opt_val_loss_prev = opt_val_loss  
        print(i)
        print(opt_val_loss)
        print(earlystop_num)
        if earlystop_num > exit_k:
            earlystop_flag = 1
                
    return earlystop_flag, earlystop_num, best_val_loss, opt_val_loss_prev
	
	
	
# Example

for i in range(iteration):
    permutation = np.random.choice(train_num,train_num,        replace=False)
    if  earlystop_flag == 1:
        break
    for j in range(batchnum):
        update_num = update_num + 1
        batch_index = permutation[(j * batch_size): (j + 1) * batch_size]
        batch_x = train_x[batch_index, :]
        batch_y = train_y[batch_index, :]
        sess.run(optimizer_f, feed_dict={x: batch_x, y_: batch_y, 'phase:0': 1})
       
       ######################## early stopping ###############################################
        earlystop_flag, earlystop_num, best_val_loss, opt_val_loss_prev = early_stopping(i, update_num, earlystop_flag, earlystop_num, best_val_loss, opt_val_loss_prev, path1)
        
        if earlystop_flag == 1:
            break 
    saver.save(sess, path1 +'/model', global_step = i)