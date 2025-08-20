import datetime
import requests
import time
from random import randrange

from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message


class InputAgent(Agent):
    class InformBehav(PeriodicBehaviour):
        async def run(self):
            print(f"PeriodicSenderBehaviour running at {datetime.datetime.now().time()}: {self.counter}")
            msg = Message(to=self.get("receiver_jid"))  # Instantiate the message
            msg.body = "Temperature " + str(self.counter)  # Set the message content

            await self.send(msg)
            print("Message sent!")

            #if self.counter == 5:
                #self.kill()
            self.counter += randrange(-1,1)

        async def on_end(self):
            # stop agent from behaviour
            await self.agent.stop()

        async def on_start(self):
            self.counter = 20

    async def setup(self):
        print(f"PeriodicSenderAgent started at {datetime.datetime.now().time()}")
        start_at = datetime.datetime.now() + datetime.timedelta(seconds=30)
        b = self.InformBehav(period=15, start_at=start_at)
        self.add_behaviour(b)

class Sector1Agent(Agent):
    class RecvBehav(CyclicBehaviour):
        async def run(self):
            print("RecvBehav running")
            msg = await self.receive(timeout=60)  # wait for a message for 10 seconds
            if msg:
                print("Message received: {}".format(msg.body))
            else:
                print("Did not received any message after 60 seconds")
                self.kill()

            API_URL = "URL"  # Ejemplo, verifica la URL real
            API_KEY = "KEY"  # Si es necesaria

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "deepseek-chat",  # o "deepseek-coder" si es para código
                "messages": [
                    {"role": "user", "content": f"el sensor de temperatura de mi panel solar tiene un rango entre 5 y 40 grados centigrados, la {msg.body}, ¿que acciones puedo tomar? genera solo comandos y se breve"}
                ],
                "temperature": 0.7
            }
            response = requests.post(API_URL, headers=headers, json=data)
            print("=============================")
            if response.status_code == 200:
                respuesta = response.json()
                print(respuesta)
                print(respuesta["choices"][0]["message"]["content"])
            else:
               print("Error:", response.status_code, response.text)


        async def on_end(self):
            await self.agent.stop()

    async def setup(self):
        print("ReceiverAgent started")
        b = self.RecvBehav()
        self.add_behaviour(b)

if __name__ == "__main__":

    sector1_jid = "Sector1Agent@07f.de"
    passwd = "asda121!asda.."
    sector1agent = Sector1Agent(sector1_jid, passwd)

    input_jid = "InputAgent@07f.de"
    passwd = "asda121!asda.."
    inoutagent = InputAgent(input_jid, passwd)

    future = sector1agent.start(auto_register=True)
    future.result()  # wait for receiver agent to be prepared.

    inoutagent.set("receiver_jid", sector1_jid)  # store receiver_jid in the sender knowledge base
    inoutagent.start(auto_register=True)

    while sector1agent.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            inoutagent.stop()
            sector1agent.stop()
            break
    print("Agents finished")
    quit_spade()
