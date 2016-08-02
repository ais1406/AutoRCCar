import numpy as np
import curses
import cv2
import socket

class CollectTrainingData(object):
    
    def __init__(self):

        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.refresh()
    
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.1.121', 8000))
        self.server_socket.listen(0)
        
        # accept a single connection 
        #self.connection = self.server_socket.accept()[0].makefile('rb')
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')


        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')
        self.collect_image()

    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print 'Start collecting images...'
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        # stream video frames one by one
        try:
            print "connect from: ",self.client_address
            stream_bytes = ' '
            frame = 1
            while True:
                #print 'start'
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    #print frame
                    # select lower half of the image
                    roi = image[120:240, :]
                    
                    # save streamed images
                    #cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), image)
                    
                    #cv2.imshow('roi_image', roi)
                    cv2.imshow('image', image)
                     
                    # reshape the roi image into one row array
                    temp_array = roi.reshape(1, 38400).astype(np.float32)
                    
                    frame += 1
                    total_frame += 1
                    
                    c = self.stdscr.getch()
                    
                    if cv2.waitKey(1) & 0xFF == ord('x'):
                        break

                    # simple orders
                    if c == curses.KEY_UP or c == ord('w') or c == ord('W'):
                    # if key_input[pygame.K_UP]:
                        print("Forward")
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[2]))
                        
                    elif c == curses.KEY_RIGHT or c == ord('d') or c == ord('D'):
                    # elif key_input[pygame.K_RIGHT]:
                        print("Right")
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[1]))
                        saved_frame += 1
                        
                    elif c == curses.KEY_LEFT or c == ord('a') or c == ord('A'):
                    # elif key_input[pygame.K_LEFT]:
                        print("Left")
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[0]))
                        saved_frame += 1
                        
                    elif c == ord('q') or c == ord('Q'):
                    # elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print 'exit'
                        break  

            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]

            # save training data as a numpy file
            np.savez('training_data_temp/test08.npz', train=train, train_labels=train_labels)

            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print 'Streaming duration:', time0

            print(train.shape)
            print(train_labels.shape)
            print 'Total frame:', total_frame
            print 'Saved frame:', saved_frame
            print 'Dropped frame', total_frame - saved_frame
            


        finally:
            self.connection.close()
            self.server_socket.close()
            curses.endwin()

if __name__ == '__main__':
    CollectTrainingData()

