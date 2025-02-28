from torchvision.datasets import DatasetFolder
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import base_dataset
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from spikingjelly.activation_based import layer
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DVS128Gesture(base_dataset.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            frame_duration: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The DVS128 Gesture dataset, which is proposed by `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.


        .. admonition:: Note
            :class: note

            In SpikingJelly, there are 1176 train samples and 288 test samples. The total samples number is 1464.

            .. code-block:: python

                from spikingjelly.datasets import dvs128_gesture

                data_dir = 'D:/datasets/DVS128Gesture'
                train_set = dvs128_gesture.DVS128Gesture(data_dir, train=True)
                test_set = dvs128_gesture.DVS128Gesture(data_dir, train=False)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1176, test samples = 288
                # total samples = 1464


            While from the origin paper, `the DvsGesture dataset comprises 1342 instances of a set of 11 hand and arm \
            gestures`. The difference may be caused by different pre-processing methods.

            `snnTorch <https://snntorch.readthedocs.io/>`_ have the same numbers with SpikingJelly:

            .. code-block:: python

                from snntorch.spikevision import spikedata

                train_set = spikedata.DVSGesture("D:/datasets/DVS128Gesture/temp2", train=True, num_steps=500, dt=1000)
                test_set = spikedata.DVSGesture("D:/datasets/DVS128Gesture/temp2", train=False, num_steps=1800, dt=1000)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1176, test samples = 288
                # total samples = 1464


            But `tonic <https://tonic.readthedocs.io/>`_ has different numbers, which are close to `1342`:

            .. code-block:: python

                import tonic

                train_set = tonic.datasets.DVSGesture(save_to='D:/datasets/DVS128Gesture/temp', train=True)
                test_set = tonic.datasets.DVSGesture(save_to='D:/datasets/DVS128Gesture/temp', train=False)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1077, test samples = 264
                # total samples = 1341


            Here we show how 1176 train samples and 288 test samples are got in SpikingJelly.

            The origin dataset is split to train and test set by ``trials_to_train.txt`` and ``trials_to_test.txt``.


            .. code-block:: shell

                trials_to_train.txt:

                    user01_fluorescent.aedat
                    user01_fluorescent_led.aedat
                    ...
                    user23_lab.aedat
                    user23_led.aedat

                trials_to_test.txt:

                    user24_fluorescent.aedat
                    user24_fluorescent_led.aedat
                    ...
                    user29_led.aedat
                    user29_natural.aedat

            SpikingJelly will read the txt file and get the aedat file name like ``user01_fluorescent.aedat``. The corresponding \
            label file name will be regarded as ``user01_fluorescent_labels.csv``.

            .. code-block:: shell

                user01_fluorescent_labels.csv:

                    class	startTime_usec	endTime_usec
                    1	80048239	85092709
                    2	89431170	95231007
                    3	95938861	103200075
                    4	114845417	123499505
                    5	124344363	131742581
                    6	133660637	141880879
                    7	142360393	149138239
                    8	150717639	157362334
                    8	157773346	164029864
                    9	165057394	171518239
                    10	172843790	179442817
                    11	180675853	187389051




            Then SpikingJelly will split the aedat to samples by the time range and class in the csv file. In this sample, \
            the first sample ``user01_fluorescent_0.npz`` is sliced from the origin events ``user01_fluorescent.aedat`` with \
            ``80048239 <= t < 85092709`` and ``label=0``. ``user01_fluorescent_0.npz`` will be saved in ``root/events_np/train/0``.





        """
        assert train is not None
        super().__init__(root, train, data_type, frames_number, frame_duration, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'
        return [
            ('DvsGesture.tar.gz', url, '8a5c71fb11e24e5ca5b11866ca6c00a1'),
            ('gesture_mapping.csv', url, '109b2ae64a0e1f3ef535b18ad7367fd1'),
            ('LICENSE.txt', url, '065e10099753156f18f51941e6e44b66'),
            ('README.txt', url, 'a0663d3b1d8307c329a43d949ee32d19')
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract [{fpath}] to [{extract_root}].')
        extract_archive(fpath, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''
        return base_dataset.load_aedat_v3(file_name)

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = DVS128Gesture.load_origin_data(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # read csv file and get time stamp and label of each sample
        # then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
        label_file_num = [0] * 11

        # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
        # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
        # are replaced by new codes.
        for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np.savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1
            print("更新")


        '''for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            #num_events = len(events['t'])
            #noise_t = np.random.normal(0, 0.01, num_events)
            #noise_x = np.random.randint(-1, 2, num_events)  # Small shifts in x
            #noise_y = np.random.randint(-1, 2, num_events)  # Small shifts in y
    
            events['t'] = events['t']+ np.random.normal(0, 1, size=events['t'].shape)
            x_shift = np.random.randint(-2, 2)
            y_shift = np.random.randint(-2, 2)
            events['x'] = np.clip(events['x'] + x_shift, 0, 127)
            events['y'] = np.clip(events['y'] + y_shift, 0, 127)
            angle_rad = np.radians(np.random.uniform(-10, 10))
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            x_center, y_center = 64, 64  # Center of the frame (128x128)

            # Shift the origin to the center
            x_shifted = events['x'] - x_center
            y_shifted = events['y'] - y_center

            # Apply rotation matrix
            x_new = cos_angle * x_shifted - sin_angle * y_shifted + x_center
            y_new = sin_angle * x_shifted + cos_angle * y_shifted + y_center

            events['x'] = np.clip(x_new, 0, 127).astype(np.int32)
            events['y'] = np.clip(y_new, 0, 127).astype(np.int32)
            #events['x'] = np.clip(events['x'] + noise_x, 0, 127)  # Keep within bounds
            #events['y'] = np.clip(events['y'] + noise_y, 0, 127)
            np.savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1
            print("代码跑了！！！！！！！！！！！！！！！")'''
        


    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedat_dir = os.path.join(extract_root, 'DvsGesture')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in range(11):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        with open(os.path.join(aedat_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 16)) as tpe:
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '_labels.csv'), train_dir)

                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file,
                                   os.path.join(aedat_dir, fname + '_labels.csv'), test_dir)

            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128


if __name__ == "__main__":
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from spikingjelly.clock_driven import neuron, functional
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    learning_rate = 1e-4
    train_epoch = 100
    simulation_steps = 8
    dataset_dir = './DVS128GestureDataset'
    Delta_t = 1e-9
    t0 = 11.5e-9
    tau = np.exp(-Delta_t / t0)
    T=1
    best_accuracy = 0
    best_epoch = 0
    best_predictions = None

    # Load the DVS128Gesture dataset
    dataset_train = DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=simulation_steps, split_by='number')
    dataset_test = DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=simulation_steps, split_by='number')

    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False
        )

    # SEBlock remains the same

    
    class SEBlock(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super(SEBlock, self).__init__()
            self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels, bias=False),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            b, c, _, _, _ = x.size()
            y = self.global_avg_pool(x).view(b, c)  # Squeeze step
            y = self.fc(y).view(b, c, 1, 1, 1)  # Excitation step
            return x * y  # Re-weight the input channels

# Define the neural network
    class SpikingNet(nn.Module):
        def __init__(self):
            super(SpikingNet, self).__init__()
            # Convolutional layers
            self.conv1 = layer.Conv2d(in_channels=2, out_channels=128, kernel_size=3, padding=1,step_mode='m')  # Output: [batch_size, 64, 64, 64]
            self.conv2 = layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,step_mode='m')  # Output: [batch_size, 128, 32, 32]
            self.conv3 = layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,step_mode='m') 
            self.conv4 = layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,step_mode='m') 
            self.conv5 = layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,step_mode='m') 
            #self.conv4 = nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=3, stride=2, padding=1) 
            #self.se1=SEBlock(128)
            #self.se2=SEBlock(512)
            #self.se3=SEBlock(2048)
            
            # Batch normalization for convolutional layers
            self.bn_conv1 = layer.BatchNorm2d(128,step_mode='m')
            self.bn_conv2 = layer.BatchNorm2d(128,step_mode='m')
            self.bn_conv3 = layer.BatchNorm2d(128,step_mode='m')
            self.bn_conv4 = layer.BatchNorm2d(128,step_mode='m')
            self.bn_conv5 = layer.BatchNorm2d(128,step_mode='m')
            #self.bn_conv4 = nn.BatchNorm3d(2048)
            # Fully connected layers
            self.pool = layer.MaxPool2d(2, 2,step_mode='m') 
            
            self.fc1 = nn.Linear(512*4*4*2, 1024, bias=False)  # Adjusted input size after conv layers
            self.neuron1 = neuron.MTJ(tau=tau, v_threshold=450., v_reset=300.)
            self.bn1 = nn.LayerNorm(1024)
            self.dropout1 = nn.Dropout(p=0.5)
    
            self.fc2 = nn.Linear(1024, 256, bias=False)  # Adjust this layer as needed
            self.neuron2 = neuron.MTJ(tau=tau, v_threshold=450., v_reset=300.)
            self.bn2 = nn.BatchNorm1d(256)
            self.dropout2 = nn.Dropout(p=0.5)
    
            self.fc3 = nn.Linear(256, 11, bias=True)  # Final layer (output size 11)
            
            #self.neuron3 = neuron.MTJ(tau=tau, v_threshold=370., v_reset=300.)
           # self.neuron4 = neuron.MTJ(tau=tau, v_threshold=370., v_reset=300.)
            #self.neuron5 = neuron.MTJ(tau=tau, v_threshold=370., v_reset=300.)
            #self.neuron6 = neuron.MTJ(tau=tau, v_threshold=370., v_reset=300.)
            #self.neuron7 = neuron.MTJ(tau=tau, v_threshold=370., v_reset=300.)
        def forward(self, x):
            #batch_size, time_steps, channels, height, width = x.shape  # Unpack the input shape
            # Collapse batch_size and time_steps into one dimension
           # x = x.view(batch_size * time_steps, channels, height, width)  # Reshape to [batch_size*time_steps, channels, height, width]
    
            # Convolutional layers with activation and batch normalization
            x = self.pool(F.relu((self.bn_conv1(self.conv1(x)))))  # Conv1 + BatchNorm + ReLU
            x = self.pool((F.relu(self.bn_conv2(self.conv2(x))))) # Conv2 + BatchNorm + ReLU
            x = self.pool(F.relu(self.bn_conv3(self.conv3(x))))
            x = self.pool(F.relu(self.bn_conv4(self.conv4(x))))
            x = self.pool(F.relu(self.bn_conv5(self.conv5(x))))
            #x = F.relu(self.bn_conv4(self.conv4(x)))
            # Flatten the feature maps from conv layers
            x = x.view(x.size(0), -1)  # Flatten: [batch_size*time_steps, 128 * 32 * 32]
            
            # Fully connected layers
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.neuron1(x)  # Spiking layer
            x = self.dropout1(x)
            
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.neuron2(x)  # Spiking layer
            #x = self.dropout2(x)
            
            x = self.fc3(x)  # Output layer (no activation, raw logits)
    
            # Reshape back to [batch_size, time_steps, num_classes]
            #x = x.view(batch_size, time_steps, -1)
            return x

    
    net = SpikingNet().to(device)


    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4, verbose=True)
    # Function to reset the spiking neurons' states after each forward pass
    def reset_net(net):
        for layer in net.modules():
            if hasattr(layer, 'reset'):
                layer.reset()
                
                # Training loop
    train_accuracies = []
    test_accuracies = []
    for epoch in range(train_epoch):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for event_reprs, target in tqdm(train_data_loader, desc=f'Training Epoch {epoch+1}/{train_epoch}'):
            event_reprs, target = event_reprs.to(device), target.to(device)
            # Forward pass
            optimizer.zero_grad()
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net((event_reprs.float()))
                    #out_spikes_counter = net(frequence_code_spiking(img).float())
                else:
                    out_spikes_counter += net((event_reprs.float()))
                    #out_spikes_counter = net(frequence_code_spiking(img).float())

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T
            #print(out_spikes_counter_frequency)
            loss = criterion(out_spikes_counter_frequency, target)

            correct += (out_spikes_counter_frequency.max(1)[1] == target.to(device)).float().sum().item()
            total += target.size(0)
            #print(out_spikes_counter_frequency.max(1)[1],target)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
                    
            # Reset the spiking neuron states
            reset_net(net)
                    
            running_loss += loss.item()
        accuracy = correct / total * 100
        train_accuracies.append(accuracy)
        print(f'Epoch {epoch+1}/{train_epoch}, Loss: {running_loss/len(train_data_loader)}, accuracy:{accuracy}')
        scheduler.step(accuracy)        
        # Testing loop
        net.eval()
        correct = 0
        total = 0
        total_pred = []
        total_target = []
        with torch.no_grad():
            for event_reprs, target in tqdm(test_data_loader, desc='Testing'):
                event_reprs, target = event_reprs.to(device), target.to(device)
                #outputs = net(event_reprs.float())
                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(event_reprs.float())
                        #out_spikes_counter = net(frequence_code_spiking(img).float())
                    else:
                        out_spikes_counter += net(event_reprs.float())
                        #out_spikes_counter = net(frequence_code_spiking(img).float())
                out_spikes_counter_frequency = out_spikes_counter / T
                #print(out_spikes_counter_frequency)
                #print(out_spikes_counter_frequency.max(1)[1].shape)
                #print(target.shape)
                correct += (out_spikes_counter_frequency.max(1)[1] == target.to(device)).float().sum().item()
                #_, predicted = torch.max(outputs, 1)
                #print(predicted)
                total += target.size(0)
                #correct += (predicted == target).sum().item()
                total_pred.extend(out_spikes_counter_frequency.max(1)[1].cpu())
                total_target.extend(target.cpu())
                
            accuracy = correct / total * 100
            test_accuracies.append(accuracy)
            print(f'Accuracy on test set: {accuracy:.2f}%')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                best_predictions = total_pred
    print(f"Best Epoch: {best_epoch + 1}, Accuracy: {best_accuracy:.2f}")

    # 计算并绘制最佳 epoch 的混淆矩阵
    conf_matrix = confusion_matrix(total_target, best_predictions)
    class_names = ['0','1', '2', '3', '4','5','6','7','8','9','10']  # 根据实际类别替换

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    plt.title(f"Confusion Matrix at Best Epoch {best_epoch + 1} with Accuracy {best_accuracy:.2f}")
    plt.savefig("DVS-matrix.svg", format="svg")
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, train_epoch + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, train_epoch + 1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.savefig("DVS.svg", format="svg")
    plt.show()                       
    import csv

    # 定义文件名
    output_file = 'accuracy_log.csv'
    
    # 写入CSV文件
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["Epoch", "Train Accuracy (%)", "Test Accuracy (%)"])
        
        # 写入每个 epoch 的准确率
        for epoch in range(train_epoch):
            writer.writerow([epoch + 1, train_accuracies[epoch], test_accuracies[epoch]])
    
    print(f'Accuracy log saved to {output_file}')



    # Test loading DVS 128 gesture dataset and spliting each sample into abritrary number of frames
    # such that each frame has about the same duration for instance 3e5 micro second
    print("Loading data - Example mode 2")
'''
    dataset_train = DVS128Gesture(root=dataset_dir, train=True, data_type='frame', split_by='frame_duration', frame_duration=300000)
    dataset_test = DVS128Gesture(root=dataset_dir, train=False, data_type='frame', split_by='frame_duration', frame_duration=300000)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    print("Creating data loaders")
    # Collate function is needed because each sample may have a different size
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, collate_fn=base_dataset.pad_seq,
        shuffle=True, num_workers=workers, pin_memory=False)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, collate_fn=base_dataset.pad_seq,
        shuffle=False, num_workers=workers, pin_memory=False)

    # Suppose we want to measure length of event representation 
    train_repr_lens = []
    for event_reprs, repr_lens, target in tqdm(data_loader, desc='Loading training data'):
        event_reprs = event_reprs
        target = target
        # Collecting length of event representation when splitting by this method
        train_repr_lens.extend(list(repr_lens))
    train_repr_lens = torch.as_tensor(train_repr_lens)
    # Print statistic of the event representation length 
    print(torch.min(train_repr_lens), torch.max(train_repr_lens), torch.mean(train_repr_lens.float()), torch.std(train_repr_lens.float()))

    # Repeat the same thing with test set 
    test_repr_lens = []
    for event_reprs, repr_lens, target in tqdm(data_loader_test, desc='Loading testing data'):
        # Do something
        event_reprs = event_reprs
        target = target
        event_reprs = event_reprs.float()
        test_repr_lens.extend(list(repr_lens))
    test_repr_lens = torch.as_tensor(test_repr_lens)
    # Print statistic of the event representation length 
    print(torch.min(test_repr_lens), torch.max(test_repr_lens), torch.mean(test_repr_lens.float()), torch.std(test_repr_lens.float()))
'''
