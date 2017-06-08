import numpy as np
import sigmoid as s
import sigmoidGradient as sg
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
num_labels, X, y, lambda_reg):
    
    
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],\
    (hidden_layer_size,input_layer_size+1),order='F')
    
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],\
    (num_labels,hidden_layer_size+1),order='F')
    
    m = len(X)
    J = 0;
    
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    
    X = np.column_stack((np.ones((m,1)),X))
    
    a2 = s.sigmoid(np.dot(X,Theta1.T))
    
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
    
    a3 = s.sigmoid( np.dot(a2,Theta2.T) )
    
    labels =  y
    
    y = np.zeros((m,num_labels))
    
    for i in xrange(m):
        y[i,labels[i]-1] = 1
    
    cost = 0
    
    for i in xrange(m):
        cost += np.sum(y[i]*np.log(a3[i]) + (1-y[i]) * np.log(1-a3[i]))
    
    J = -(1.0/m)*cost
    
    sum0fTheta1 = np.sum(np.sum(Theta1[:,1]**2))
    
    sum0fTheta2 = np.sum(np.sum(Theta2[:,1]**2))
    
    J = J + ( (lambda_reg/(2.0*m)) * (sum0fTheta1+ sum0fTheta2))
    
    
    bigDelta1 = 0
    bigDelta2 = 0
    
    for t in xrange(m):
        x= X[t]
        
        a2 = s.sigmoid( np.dot(x,Theta1.T))
    
        a2 = np.concatenate((np.array([1]),a2))
       
        a3 = s.sigmoid( np.dot(a2,Theta2.T) )
    
        delta3 = np.zeros((num_labels))
       
        for k in xrange(num_labels):
            y_k = y[t,k]
            delta3[k] = a3[k] - y_k
    
        delta2 = (np.dot (Theta2[:,1:].T, delta3).T) * sg.sigmoidGradient( np.dot(x,Theta1.T) )
    
        bigDelta1 += np.outer(delta2, x)
        bigDelta2 += np.outer(delta3,a2)
    
    Theta1_grad = bigDelta1 / m
    Theta2_grad = bigDelta2 / m
    
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    
    Theta1_grad += (float(lambda_reg)/m)*Theta1
    Theta2_grad += (float(lambda_reg)/m)*Theta2

    print Theta1_grad.shape
    print Theta2_grad.shape
    print Theta1_grad

    Theta1_grad[:,0] = Theta1_grad_unregularized[:,0]
    Theta2_grad[:,0] = Theta2_grad_unregularized[:,0]

    print Theta1_grad

    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))
     
    return J,grad
