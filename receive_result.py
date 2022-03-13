from kafka import KafkaConsumer, KafkaProducer, errors
import json
import argparse


class ResultReceiver:
    def __init__(self, kafka_server, kafka_topic, output_file, interval):
        self.kafka_server = kafka_server
        self.kafka_topic = kafka_topic
        self.output_file = output_file
        self.interval = interval
        self.consumer = KafkaConsumer(self.kafka_topic, bootstrap_servers=self.kafka_server)

    def start_receive(self):
        stream_file = open(self.output_file, 'w')
        flag = 0
        frame_number = 0
        while True:
            raw_messages = self.consumer.poll(
                timeout_ms=3000.0, max_records=5000)
            if len(raw_messages.items()) == 0:
                flag += 1
            for _, msg_list in raw_messages.items():
                for msg in msg_list:
                    msg_value = json.loads(msg.value.decode('utf-8'))
                    for i in range(self.interval + 1):
                        for bbox in msg_value['bbox']:
                            if bbox['label'] == 'person':
                                label_num = 1
                            elif bbox['label'] == 'car' or bbox['label'] == 'trunk':
                                label_num = 2
                            else:
                                continue
                            stream_file.write('{:d} {:d} {:d} {:d} {:d} {:d} {:.6f}\n'.format(frame_number + i, bbox['topleftx'], bbox['toplefty'], bbox['bottomrightx'], bbox['bottomrighty'], label_num, bbox['score']))
                    frame_number += 1
                    if frame_number % 1000 == 0:
                        print(frame_number)
            if flag >= 5:
                break
        stream_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topic", help="kafka topic name",
                        default="experiment-jetson")
    parser.add_argument("-o", "--output", help="output file name",
                        default="output.txt")
    parser.add_argument("-f", "--fps", help="fps",
                        default="30")
    args = parser.parse_args()
    default_fps = 30
    fps = int(args.fps)
    interval = int(default_fps / fps) - 1
    result_receiver = ResultReceiver('172.16.29.105:9092', args.topic, args.output, interval)
    result_receiver.start_receive()


if __name__ == "__main__":
    main()
