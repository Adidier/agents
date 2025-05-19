/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ruleasm;

import jade.core.AID;
import jade.core.behaviours.*;
import jade.domain.FIPAAgentManagement.*;
import jade.lang.acl.*;
import jade.lang.acl.ACLMessage;
import jade.domain.FIPAAgentManagement.*;
import jade.lang.acl.MessageTemplate;
import jade.gui.*;

/**
 *
 * @author Virgilio
 */
public class Administrador extends GuiAgent
{

    Principal mainFrame;
    public static String numr = null,  motor = null,  tipo_V = null,
            num_P = null,  size = null, volar=null, motorTipo=null;
    public String tipoVA;

    @Override
    public void setup()
    {
        mainFrame = new Principal(this);
        (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "<Administrador> Activo");
        addBehaviour(new RecibeListoC());
        addBehaviour(new RecibeListoA());
        addBehaviour(new RecibeListoE());


    }

    @Override
    protected void onGuiEvent(GuiEvent arg0)
    {
        int ev = arg0.getType();
        switch (ev)
        {
            case 1:
                numr = mainFrame.NumR;
                motor = mainFrame.Motor;
                addBehaviour(new TipoVBehaviour());
                addBehaviour(new getTipoVCBehaviour());
                addBehaviour(new setTipoVABehaviour());
                addBehaviour(new getTipoVEBehaviour());
                break;
            case 2:
                numr = mainFrame.NumR;
                motor = mainFrame.Motor;
                tipo_V = mainFrame.tipoV;
                num_P = mainFrame.numP;
                size = mainFrame.size;
                tipoVA = "Auto";
                addBehaviour(new getVehiculo());
                addBehaviour(new VehiculoA());
                break;
            case 3:
                numr = mainFrame.NumR;
                motor = mainFrame.Motor;
                tipo_V = mainFrame.tipoV;
                tipoVA = "Cyclic";
                addBehaviour(new getVehiculo());
                addBehaviour(new VehiculoC());
                break;
        }
    }
}

class RecibeListoC extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.INFORM);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Cyclic", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Coordinador> Recibe informe de listo de Agente Cyclic");
        }
    }
}

class RecibeListoE extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.INFORM);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Aereo", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Coordinador> Recibe informe de listo de Agente Aereo");
        }
    }
}

class RecibeListoA extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.INFORM);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Auto", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Coordinador> Recibe informe de listo de Agente Automovil");
        }
    }
}

class TipoVBehaviour extends OneShotBehaviour
{

    @Override
    public void action()
    {
        ACLMessage msg = new ACLMessage(ACLMessage.REQUEST);
        msg.addReceiver(new AID("Cyclic", AID.ISLOCALNAME));
        msg.setContent("TipoVehiculo");
        ((Administrador) myAgent).send(msg);
        (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Coordinador> Envia peticion de Tipo de vehiculo a Agente Cyclic");
        ACLMessage msg2 = new ACLMessage(ACLMessage.REQUEST);
        msg2.addReceiver(new AID("Auto", AID.ISLOCALNAME));
        msg2.setContent("TipoVehiculo");
        ((Administrador) myAgent).send(msg2);
        (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Coordinador> Envia peticion de Tipo de vehiculo a Agente Auto");
        ACLMessage msg3 = new ACLMessage(ACLMessage.REQUEST);
        msg3.addReceiver(new AID("Aereo", AID.ISLOCALNAME));
        msg3.setContent("TipoVehiculo");
        ((Administrador) myAgent).send(msg3);
        (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Coordinador> Envia peticion de Tipo de vehiculo a Agente Aereo");

    }
}

class getTipoVCBehaviour extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.PROPOSE);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Cyclic", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() +
                    "\n<Coordinador> Recibe propuesta --" + msg4.getContent() + " de Agente Cyclic");
            if (msg4.getContent() != null)
            {
                System.out.println("Tipo Vehiculo" + msg4.getContent());
                ((Administrador) myAgent).mainFrame.lanzaFrame(msg4.getContent());
            }
        }
    }
}

class getTipoVEBehaviour extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.PROPOSE);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Aereo", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() +
                    "\n<Coordinador> Recibe propuesta --" + msg4.getContent() + " de Agente Aereo");
            if (msg4.getContent() != null)
            {
                System.out.println("Tipo Vehiculo" + msg4.getContent());
                ((Administrador) myAgent).mainFrame.lanzaFrame(msg4.getContent());
            }
        }
    }
}

class setTipoVABehaviour extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.PROPOSE);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Auto", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() +
                    "\n<Coordinador> Recibe propuesta --" + msg4.getContent() + " de Agente Auto");
            if (msg4.getContent() != null)
            {
                System.out.println("Tipo Vehiculo" + msg4.getContent());
                ((Administrador) myAgent).mainFrame.lanzaFrame(msg4.getContent());
            }        
        }
    }
}

class getVehiculo extends OneShotBehaviour
{

    @Override
    public void action()
    {
        ACLMessage msg = new ACLMessage(ACLMessage.REQUEST_WHEN);
        msg.addReceiver(new AID(((Administrador) myAgent).tipoVA, AID.ISLOCALNAME));
        msg.setContent("TipoVehiculo");
        ((Administrador) myAgent).send(msg);
    }
}

class VehiculoA extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.CONFIRM);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Auto", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg != null)
        {
            ((Administrador) myAgent).mainFrame.lanzaVehiculo(msg.getContent());
        }
    }
}

class VehiculoC extends OneShotBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.CONFIRM);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Cyclic", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg = ((Administrador) myAgent).blockingReceive(tem3);
        if (msg != null)
        {
            ((Administrador) myAgent).mainFrame.lanzaVehiculo(msg.getContent());
        }
    }
}
