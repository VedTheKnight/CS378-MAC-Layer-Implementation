import numpy as np
import pyaudio 
import keyboard
import random
import time
from time import sleep

mac_id = 3
mac_id_dict = {0:'00',1: '01',2:'10',3:'11'}
frequencies = {'2': 500, '1': 4000, '0':1000, '3':750, '4': 2000} 

# Function to xor two strings
def xor(a, b):
    answer = ""
    for i in range(0, len(b)):
        if a[i] == b[i]:       # If the two bits are equal then xor value is zero 
            answer += "0" 
        else:                  # Else xor is one 
            answer += "1"
    return answer

# Function to execute long division in the CRC method
def division(dividend, divisor):
    if len(dividend) < len(divisor): # If dividend is smaller than divisor then the remainder is simply the dividend, for completeness
        remainder = '0'*(len(divisor)-len(dividend)) # Just to keep total same number of bits as that of divisor
        remainder += dividend
        return remainder
    else:
        ptr = len(divisor)  # Pointer for the last bit in the dividend that needs to be divided
        curr = dividend[0:len(divisor)] # The current remainder to be xor'ed with the divisor
        while ptr < len(dividend):
            # If the first bit is one, then xor without first and add the last bit to the current remainder
            if curr[0] == '1':
                curr = xor(divisor, curr)
                curr = curr[1:] # Remove the first bit
                curr += dividend[ptr]
            # If first bit is zero, then simply ignore the bit and add the last bit to the current remainder
            else:
                curr = curr[1:]
                curr += dividend[ptr]
            ptr += 1 # Move pointer ahead
        # Doing the same for the final remainder, when the pointer is beyond the last bit of the dividend
        if curr[0] == '1':
            curr = xor(divisor, curr)
            curr = curr[1:] # Remove the first bit in remainder
        else:
            curr = curr[1:] # Simply remove first bit if first bit is zero
        return curr

# Function to encode the data using CRC
def encodeData(data, key):
    # Add zeros and do long division
    remainder = division(data + '0'*(len(key)-1), key)
    # Appending the remainder to the original data
    return (data + remainder)

# Function that returns true if the remainder is all zeros (that is, in our original space), otherwise false
def decodeData(data, key):
    remainder = division(data, key)
    reqd_remainder = '0'*(len(key)-1)
    return remainder == reqd_remainder

def take_input():
    messages = []

    msg1 = input()
    msg2 = input()

    message1, destination1 = msg1.split(' ')
    message2, destination2 = msg2.split(' ')

    if(destination1 != "-1"):
        messages.append([message1,destination1])
    if(destination2 != "-1"):
        messages.append([message2,destination2])
    return messages


# The message is then processed to get rid of the '2's which act as boundaries for the bits in the actual message
def process(message_bits):
    result = message_bits[0]
    for i in range(1, len(message_bits)):
        if message_bits[i] != message_bits[i - 1] or message_bits[i] == '3': 
            result += message_bits[i]

    result = result.replace('2', '')
    return result

# Converts the audio signal captured from the sender into a bitstring
def decode_audio_to_message(audio_signal, sample_rate=44100):
    duration = 0.1 # Duration of each bit sound in seconds (The receiver recieves each bit twice to avoid errors)
    threshold = 2  # Threshold to figure out the nearest matching frequency
    frequency_0 = frequencies['0']
    frequency_1 = frequencies['1']
    frequency_2 = frequencies['2']
    frequency_3 = frequencies['3']

    num_samples_per_bit = int(sample_rate * duration)
    message_bits = "2"
    for i in range(0, len(audio_signal), num_samples_per_bit):
        # The dominant frequency used to generate this wave is found through fast fourier transform
        bit_signal = audio_signal[i:i + num_samples_per_bit]
        fft_result = np.fft.fft(bit_signal)
        freq = np.fft.fftfreq(len(bit_signal), d=1/sample_rate)
        peak_freq = abs(freq[np.argmax(np.abs(fft_result))])

        # The bit corresponding to the dominant frequency is then added to the message if it is valid and discarded if the 
        # frequency is noise
        if abs(peak_freq - frequency_0) < threshold:
            message_bits += '0'
            
        elif abs(peak_freq - frequency_1) < threshold:
            message_bits += '1'

        elif abs(peak_freq - frequency_2) < threshold:
            message_bits += '2'
        
        elif abs(peak_freq - frequency_3) < threshold:
            message_bits += '3'

    message_bits = process(message_bits)
    return message_bits

def decode_acknowledgement(audio_signal, sample_rate = 44100):
    duration = 0.1 # Duration of each bit sound in seconds (The receiver recieves each bit twice to avoid errors)
    threshold = 2  # Threshold to figure out the nearest matching frequency

    frequency_4 = frequencies['4']
    message_bits = ''
    num_samples_per_bit = int(sample_rate * duration)

    for i in range(0, len(audio_signal), num_samples_per_bit):
        # The dominant frequency used to generate this wave is found through fast fourier transform
        bit_signal = audio_signal[i:i + num_samples_per_bit]
        fft_result = np.fft.fft(bit_signal)
        freq = np.fft.fftfreq(len(bit_signal), d=1/sample_rate)
        peak_freq = abs(freq[np.argmax(np.abs(fft_result))])

        if abs(peak_freq - frequency_4) < threshold:
            message_bits += '4'

    return message_bits

def generate_waveform(bit_string, sample_rate=44100, duration=0.2):
    waveform = np.array([])
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    for bit in bit_string:
        frequency = frequencies[bit]
        wave = np.sin(frequency * 2 * np.pi * t)
        waveform = np.concatenate((waveform, wave))

    return waveform

def play_waveform(waveform, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
    stream.write(waveform.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()

def encode_message(bit_string):
    encoded_bit_string = "22222" 
    for i in range(0, len(bit_string)):
        encoded_bit_string+=bit_string[i]+"2"
    return encoded_bit_string
""" 
    We bitstuff our message as follows:
    Everytime '1111' occurs in a message we replace it with '11110'.
    The end of a message is denoted by '0111110'. To make sure that this string does not occur anywhere we 
    do bitstuffing.
    This functio is called at the sender's end.
"""
def bitstuff(msg):
    bmsg = ''
    for i in range(len(msg)):
        bmsg += msg[i]
        if (len(bmsg) >= 4):
            if (bmsg[len(bmsg) - 4:] == '1111'):
                bmsg += '0'
    return bmsg
"""
    This function reverses the effects of bitstuffing. 
    Whenever a substring '1111' is encountered, the immediate bit is dropped as it is an extra '0'.
    This is called at the receiver's end.
"""
def ibitstuff(bmsg):
    msg = ''
    i = 0
    while (i < len(bmsg)):
        msg += bmsg[i]
        if (i >= 3):
            if (bmsg[i - 3:i + 1] == '1111'):
                i += 1
        i += 1
    return msg
"""
    The stream of '3's at the beginning act like a preamble and helps in synchronising the clocks of the sender
    and receiver(s). This ensures that the beginning of the message is not missed.
    The structure of the message is as follows: 
            |PREAMBLE|DEST_MAC_ID|SRC_MAC_ID|MSG|CRC|END_MARKER|
    Except for preamble and end_marker the rest of the segments are bitstuffed.
"""
def get_payload(message, target):
    message = encodeData(mac_id_dict[target]+mac_id_dict[mac_id]+message, '101011')
    payload = '333'+bitstuff(message)+'011111'
    return payload
"""
    pyaudio code to play the final message and display the time of dispatch.
"""
def transmit_message(message, target):
    payload = get_payload(message, target)
    print(f"[SENT] {message} {target} {time.strftime('%H:%M:%S')}")
    waveform = generate_waveform(encode_message(payload))
    play_waveform(waveform)
"""
    This function converts the binary representation of an integer to a string of size 5
"""
def number_to_bitstring(num):
    bitstring = bin(num)[2:]
    return bitstring.zfill(5)
"""
    The acknowledgment sent by the receiver is the string '4444'.
"""
def generate_acknowledgement(sender, receiver):
    return '4444'
"""
    pyaudio code to transmit the acknowledgement
"""    
def transmit_ack(ack):
    waveform = generate_waveform(encode_message(ack))
    play_waveform(waveform)
"""
    This function contains the core logic for the MAC layer
"""
def run():
    messages = take_input()
    ctr = [0] 
    message_queue = []
    """
        This function handles keyboard interrupts.
        Upon pressing the enter key, the messages from the input files are read by the sender and pushed onto
        to it's message queue.
    """
    def on_key_event(e):
        if e.name == 'enter':
            print('Enter key was pressed!')
            if(len(messages) > ctr[0]):
                message_queue.append([messages[ctr[0]][0], int(messages[ctr[0]][1])])
                ctr[0]+=1

    keyboard.on_press(on_key_event)
    
    #start continuous carrier sensing

    p = pyaudio.PyAudio()
    # The audio message is received through functions from pyaudio module
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    print(f"Starting Carrier Sense {mac_id}")
    frames = [] 
    empty_carrier_counter = 0 # counts the number of frames for which the channel has been silent 
    previous_message_bits = '' # stores the obtained message in the previous frame
    state = 'CS' # stores the state of the node at each instant
    message_sender = '' # stores who the sender of the message is  
    wack_ctr = 0 # counts how long the node is waiting for wack
    i = 0
    r = -1
    cs_counter=0
    cb_counter=0
    exponential_upper_bound = 200
    alpha = 2
    exponential_cutoff = 1000
    message_dropped = False
    """
        States: 
            1.CS     : Carrier Sensing      - The Node checks if the channel is idle or not for 30 iterations.
            2.AM     : Attempt Message      - The Node has completed carrier sensing and finds the carrier empty so attempts to send its message
            3.CB     : Channel Busy         - The Node has sensed someone else's transmissions which was not meant for it so is now 
                                              staying silent for the duration of the transmissions. 
            4.WACK   : Wait Acknowledgement - The Node has sent a message and waits for an acknowledgement from 
                                              receiver for 120 iterations. After 120 iterations, if it still has not 
                                              received a response it goes to the state CS and the cycle starts again. 
                                              If it receives an acknowledgement, then it pops the message off of the
                                              message_queue and goes to the state CS.
            5.SACK   : Send Acknowledgement - The Node has to send an acknowledgement to the send after receiving 
                                              the message. Once this is done it goes back to the state CS.
            6.BSACK  : Broadcast SACK       - The Node has the minimum MAC ID among the listener nodes of the broadcast so it has to send an acknowledgement 
                                              to the sender after receiving the message. Once this is done it goes back to the state CS.
            7.WASACK : Wait and SACK        - The Node does not have the minimum MAC ID among the listener nodes of the broadcast so it has to 
                                              wait for the and listen to the minimum MAC IDs ACK. If it hears its ACK, then it also sends an acknowledgement 
                                              to the sender and goes back to the state CS, else it would simply go to the state CS.
            8.BWACK  : Broadcast WACK       - The Node has sent a message and waits for an acknowledgement from the minimum MAC
                                              receiver for 120 iterations. After 120 iterations, if it still has not 
                                              received a response it goes to the state CS and the cycle starts again. 
                                              If it receives an acknowledgement, then it goes to the state WACK to recieve ACK from the higher
                                              MAC ID reciever.
    """ 
    while True:

        # default behaviour for listening
        if(state == 'CS'):
    
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
            audio_signal = np.concatenate(frames)
            message_bits = decode_audio_to_message(audio_signal)
            """
                message_bits contains all the data seen from the time of last clearing it.
                relevant_message_bits contains only those bits that belong to the current message that is being
                transmitted in the channel.
            """
            relevant_message_bits = message_bits[i:]

            if('3' in relevant_message_bits):
                message_dropped = False #that means we have started receiving a new transmission, so now set the flag to False
                relevant_message_bits = relevant_message_bits[relevant_message_bits.find('3'):]
            relevant_message_bits = relevant_message_bits.replace('3', '')
           
            # If we do not have any message to transmit we need not maintain a count for how long the channer is idle
            if(len(message_queue) == 0):
                empty_carrier_counter = 0

            # This is done to check if channel is idle for a sufficiently long time. If yes, then message_bits is cleared
            if(message_bits==previous_message_bits):
                cs_counter=cs_counter+1
            else: 
                cs_counter=0

            # If the channel has been idle long enough (30 iterations) the frames are cleared and so is message_bits    
            if(cs_counter==30):
                empty_carrier_counter = 0
                frames=[]
                del(audio_signal)
                i=0
                cs_counter=0
                previous_message_bits=""
                message_bits=""
                message_dropped = True # Any incompletely heard message or acknowledgement is dropped
            
                
            if(len(relevant_message_bits) >= 10 and relevant_message_bits[-6:] == '011111'): # detects a message
                # Bitstuffing is first reversed 
                message = ibitstuff(relevant_message_bits[:-6])
                """
                    Since the nodes use CRC, the receiver can check if the message received is fully correct.
                    If not, it is dropped and the node goes back to carrier sensing.
                    If yes, it changes state to 'SACK' if the message is meant for it ie Send Acknowledgement and prints the received message first.
                    If the message is meant for some other node, it goes to the state 'CB' ie Channel Busy and 
                    waits for the ongoing transmission to terminate.
                """
                if not decodeData(message, '101011'):
                    message_dropped = False
                    state = 'CS'
                    empty_carrier_counter = 0
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter=0
                    previous_message_bits=""
                    message_bits=""
                    continue
             
                message = message[:-5]
                
                if(message_dropped):
                    
                    message_dropped = False
                    state = 'CS'
                    empty_carrier_counter = 0
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter=0
                    previous_message_bits=""
                    message_bits=""
                    continue

                if(message[0:2] == mac_id_dict[mac_id]): # if meant for us
                    print(f"[RECVD] {message[4:]} {int(message[2])*2+int(message[3])} {time.strftime('%H:%M:%S')}")
                    i += len(relevant_message_bits)
                    message_sender = message[2:4] 
                    message_dropped = False
                    state = 'SACK' # send acknowledgement
                    cs_counter=0
                elif(message[0:2] == '00'): # broadcast flag is set to true, everyone who has heard the message will give acknowledgement
                    print(f"[RECVD] {message[4:]} {int(message[2])*2+int(message[3])} {time.strftime('%H:%M:%S')}")
                    i += len(relevant_message_bits)
                    message_sender = message[2:4] 
                    message_dropped = False
                    state = 'BSACK' # send acknowledgement
                    cs_counter=0
                else:
                    i += len(relevant_message_bits)
                    state = 'CB' # clearing state, where there is an ongoing communication between two nodes which we want to delete
                    message_dropped = False
                    empty_carrier_counter = 0
                    cs_counter=0

            if(previous_message_bits == message_bits):
                empty_carrier_counter+=1
                if(empty_carrier_counter == 30):
                    r = random.randint(1, exponential_upper_bound)
                    state = 'AM' # attempt message state
            else:
                empty_carrier_counter = 0
                    
        elif(state == 'AM'):
            
            # we carrier sense when we are in this attempt message loop so that we know that no one else is transmitting before us
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
            audio_signal = np.concatenate(frames)
            message_bits = decode_audio_to_message(audio_signal)

            relevant_message_bits = message_bits[i:]
            """
                If there are some bits in the channel, it implies there is some ongoing transmission and the node
                goes back to carrier sensing.
            """
            if(len(relevant_message_bits) > 0):
                state = 'CS'
                empty_carrier_counter = 0
                frames=[]
                del(audio_signal)
                i=0
                cs_counter=0
                previous_message_bits=""
                message_bits=""
                continue
            """
                If the channel has been free for 'r' iterations, then the node transmits the message and goes 
                to a state WACK(BWACK) where it waits for acknowledement(s) from receiver(s).
            """
            r -= 1
            if(r == 0):
                transmit_message(message_queue[0][0], message_queue[0][1])
                if(message_queue[0][1] == 0): #it was a broadcast message:
                    state = 'BWACK'
                else:
                    state = 'WACK'
                i = len(message_bits) 

        # waits for acknowledgement after transmitting a message
        elif(state == 'WACK'):
            
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
            audio_signal = np.concatenate(frames)
            message_bits = decode_acknowledgement(audio_signal)

            relevant_message_bits = message_bits
            """
                If the node detects even a single '4', it senses it as an acknowledgement and goes to carrier sensing.
                All variables are reset.
                The transmitted message is popped off of the message queue.
            """
            wack_ctr += 1
            if(wack_ctr==120):
                if('4' in relevant_message_bits):
                    print("Received Acknowledgement")
                    wack_ctr = 0
                    exponential_upper_bound = 200
                    message_queue.pop(0) # this means that message was sent successfully so we can pop it from the queue
                    state = 'CS'
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter = 0
                    previous_message_bits=""
                    message_bits=""
                else:
                    # Since no acknowledgement is received this means that, there was a collision so we exponentially increase our r value range
                    exponential_upper_bound = min(int(exponential_upper_bound*alpha), exponential_cutoff)
                    wack_ctr = 0
                    state = 'CS'
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter = 0
                    previous_message_bits=""
                    message_bits=""
                    empty_carrier_counter = 0

        # waits for 2 acknowledgements after transmitting a broadcast message
        elif(state == 'BWACK'):
            
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
            audio_signal = np.concatenate(frames)
            message_bits = decode_acknowledgement(audio_signal)

            relevant_message_bits = message_bits

            wack_ctr += 1
            if(wack_ctr==120):
                if('4' in relevant_message_bits):
                    sleep(1)
                    wack_ctr = 0
                    exponential_upper_bound = 200
                    # we don't pop from message queue here since we need to wait for both acknowledgements
                    state = 'WACK'
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter = 0
                    previous_message_bits=""
                    message_bits=""
                else:
                    # Since no acknowledgement is received this means that, there was a collision so we exponentially increase our r value range
                    exponential_upper_bound = min(int(exponential_upper_bound*alpha), exponential_cutoff)
                    wack_ctr = 0
                    state = 'CS'
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter = 0
                    previous_message_bits=""
                    message_bits=""
                    empty_carrier_counter = 0
    
        # sends acknowledgement after receiving a message succesfully
        elif(state == 'SACK'):
            ack = generate_acknowledgement(mac_id_dict[mac_id], message_sender)
            transmit_ack(ack)
            state = 'CS'
            empty_carrier_counter = 0
            frames=[]
            del(audio_signal)
            i=0
            previous_message_bits=""
            message_bits=""
        
        elif(state == 'BSACK'):
            sender = int(message_sender[0])*2+int(message_sender[1])
            remaining_mac_ids = [mac_id for mac_id in [1, 2, 3] if mac_id != sender]
            if(mac_id == min(remaining_mac_ids)):
                ack = generate_acknowledgement(mac_id_dict[mac_id], message_sender)
                transmit_ack(ack)
                state = 'CS'
                empty_carrier_counter = 0
                frames=[]
                del(audio_signal)
                i=0
                previous_message_bits=""
                message_bits=""
            else:
                state = 'WASACK'
                empty_carrier_counter = 0
                frames=[]
                del(audio_signal)
                i=0
                previous_message_bits=""
                message_bits=""

        elif(state == 'WASACK'):
            
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
            audio_signal = np.concatenate(frames)

            message_bits = decode_acknowledgement(audio_signal)
            relevant_message_bits = message_bits

            wack_ctr += 1
            if(wack_ctr==120):
                if('4' in relevant_message_bits):
                    # now we send our acknowledgement and go to CS 
                    ack = generate_acknowledgement(mac_id_dict[mac_id], message_sender)
                    transmit_ack(ack)
                    wack_ctr = 0
                    exponential_upper_bound = 200
                    state = 'CS'
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter = 0
                    previous_message_bits=""
                    message_bits=""
                else:
                    # Since no acknowledgement is received this means that, there was a collision so we exponentially increase our r value range
                    exponential_upper_bound = min(int(exponential_upper_bound*alpha), exponential_cutoff)
                    wack_ctr = 0
                    state = 'CS'
                    frames=[]
                    del(audio_signal)
                    i=0
                    cs_counter = 0
                    previous_message_bits=""
                    message_bits=""
                    empty_carrier_counter = 0

        elif(state == 'CB'):
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
            audio_signal = np.concatenate(frames)
            message_bits = decode_audio_to_message(audio_signal)

            relevant_message_bits = message_bits[i:]
            """
                The node detects if the channel is idle or not and if it remains idle for a sufficiently long
                time, it clears the channel and starts carrier sensing in the state 'CS'
            """
            if(message_bits==previous_message_bits):
                cb_counter=cb_counter+1
            else: 
                cb_counter=0
            if(cb_counter==120):
                state = 'CS'
                empty_carrier_counter = 0
                frames=[]
                del(audio_signal)
                i=0
                cb_counter=0
                cs_counter = 0
                previous_message_bits=""
                message_bits=""

        previous_message_bits = message_bits    

run()