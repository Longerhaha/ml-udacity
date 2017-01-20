#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy
    errors = (predictions-net_worths)**2
    index = numpy.argsort(errors,0)
    delete_index = index[(numpy.size(index)-9):numpy.size(index):1,0]
                         
    age = numpy.delete(ages,delete_index)
    net_worth = numpy.delete(net_worths,delete_index)
    error = numpy.delete(errors,delete_index)
    #age = ages
    #net_worth = net_worths
    '''
    for i in range(9):
        age = numpy.delete( age,index[numpy.size(error)-1-i][0])
        net_worth = numpy.delete( net_worth,index[numpy.size(error)-1-i][0])
        error = numpy.delete( error,index[numpy.size(error)-1-i][0])
    cleaned_data = (age,net_worth,error)    
    '''
    cleaned_data = zip(age,net_worth,error)
    return cleaned_data
    