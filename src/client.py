import socket
import pickle
import numpy as np

from FinalCode import PCB_Isolation_Forest
# Class Client is Created

class Client:
    def __init__(self, host='localhost', port=12345):
        self.host = host #server is running on the same machine as the client, Stores the server's hostname or IP address.
        self.port = port # Default Value,Stores the server's port number.
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port)) #Establishes a connection to the server using the specified host and port.
        
        
#send_new_trees Method in the Client Class
    def send_new_trees(self, new_trees):   #The new trees that need to be sent to the server
        new_trees_data = pickle.dumps(new_trees) # Serializes the new_trees object into a byte stream using the pickle module. 
        self.send_data(new_trees_data) # Calls the send_data method to send the serialized new_trees_data to the server.

        # Receive updated model from server
        updated_model_data = self.receive_data() #Calls the receive_data method to receive the updated model data from the server.
        updated_model = pickle.loads(updated_model_data) #Deserializes the received byte stream back into a Python object using the pickle module. 
        print('Received updated model from server.')

        return updated_model

    def send_data(self, data): #Sends the provided data to the server using the client socket.
        self.client_socket.sendall(data)

    def receive_data(self):  #Continuously receives data from the server 
        data = b''
        while True:
            packet = self.client_socket.recv(4096)
            if not packet:
                break
            data += packet
        return data

    def close(self):
        self.client_socket.close() #closes the socket connection to the server. 

if __name__ == '__main__':
    client = Client()

    
    #new_trees is a list of new trees to be sent to the server
    new_trees = [] 

    # Send new trees to the server and receive the updated model
    updated_model = client.send_new_trees(new_trees)

    # Save the updated model 
    with open('updated_model.pkl', 'wb') as file:
        pickle.dump(updated_model, file)

    client.close()
