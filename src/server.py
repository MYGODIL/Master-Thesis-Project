import socket
import pickle
import numpy as np
from collections import deque
from FinalCode import PCB_Isolation_Forest

class Server:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))  # Binds the socket to the specified host and port.
        self.server_socket.listen(5)  # Listens for incoming connections.
        self.model = PCB_Isolation_Forest(size_of_sample=256, no_of_trees=10, Length_of_window=100, threshold=0.5, value_of_alpha=0.005)  # Initializes the model attribute of the server.
        self.Length_of_window = deque(maxlen=100)  # Initializes the sliding window.

    # Start the server
    def start(self):
        print('Server started, waiting for connection...')
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f'Connection from {addr}')
            self.handle_client(client_socket)  # Handles communication with the connected client.

    def handle_client(self, client_socket):
        try:
            # Receive new tree data
            new_trees_data = self.receive_data(client_socket)  # Receives new tree data from the client.
            new_trees = pickle.loads(new_trees_data)
            print(f'Received new trees: {new_trees}')

            # Update the model with new trees
            self.update_model(new_trees)

            # Send updated model back to client
            updated_model_data = pickle.dumps(self.model)  # Serializes the updated model.
            self.send_data(client_socket, updated_model_data)  # Sends the updated model back to the client.
        finally:
            client_socket.close()

    def receive_data(self, client_socket):
        data = b''
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet
        return data

    def send_data(self, client_socket, data):
        client_socket.sendall(data)

    # Integrates new trees into the existing model.
    def update_model(self, new_trees):
        # Replace trees with a pc value less than zero
        replaced_trees = 0
        for i, tree in enumerate(self.model.trees):
            if tree.pc_value < 0:
                self.model.trees[i] = new_trees[replaced_trees % len(new_trees)]
                replaced_trees += 1
                print(f'Replaced tree at index {i} with new tree.')

        # If there are remaining new trees that haven't been used, append them
        for i in range(replaced_trees, len(new_trees)):
            self.model.trees.append(new_trees[i])
            print(f'Appended new tree at index {len(self.model.trees) - 1}.')

        print('Model updated with new trees.')

if __name__ == '__main__':
    server = Server()
    server.start()
