import pika
import uuid
import json


class RpcClient(object):
    def __init__(self, queue_name):
        self.queue_name = queue_name  # Note
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=self.queue_name,
                                   properties=pika.BasicProperties(
                                       reply_to=self.callback_queue,
                                       correlation_id=self.corr_id,
                                   ),
                                   body=json.dumps(data))
        while self.response is None:
            self.connection.process_data_events()
        return self.response.decode("utf-8")


class NoAckClient:
    def __init__(self, queue_name):
        self.queue_name = queue_name
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=self.queue_name)

    def call(self, data):
        self.channel.basic_publish(
            exchange='', routing_key=self.queue_name, body=json.dumps(data))
        print(" [x] Sent Success")

    def close(self):
        self.connection.close()
