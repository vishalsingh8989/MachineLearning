"""
"""
import csv


def update_results(dataset_type, activation_fxn,nums_of_layer,learning_fxn,learning_rate,batch_size,training_steps, accuracy, total_loss, timetaken = "Not Calulated"):
    result = dataset_type + ", " + activation_fxn + ", "  + str(nums_of_layer) + " ," + learning_fxn + ", " + str(learning_rate) + ", " + str(batch_size) + ", " + str(training_steps) + ", " + str(accuracy)[:5] + " , " + str(total_loss) + "," + str((total_loss/600000.0)*100)[:5] +","+ str(timetaken)    
    with open('./results/test_results.csv', 'a') as csvfile:
        #datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)  
        #datawriter.write(result + "\n")
        csvfile.write(result + "\n")
        csvfile.close()
            