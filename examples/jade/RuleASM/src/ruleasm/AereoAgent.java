/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ruleasm;

import Rule.*;
import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.core.behaviours.OneShotBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

/**
 *
 * @author Virgilio
 */
public class AereoAgent extends Agent
{

    BooleanRuleBase br = new BooleanRuleBase("test");
    RuleVariable TipoVehiculo;
    RuleVariable Vehiculo;
    RuleVariable Num_Ruedas;
    RuleVariable Motor;
    RuleVariable Volar;
    RuleVariable TipoMotor;
    String resultado;

    @Override
    public void setup()
    {
        addBehaviour(new ListoE());
        addBehaviour(new TipoVE());
        addBehaviour(new getVehiculoE());
    }
    
    public String getTipoV()
    {
        BaseReglas();
        Motor.setValue(Administrador.motor);
        Volar.setValue(Administrador.volar);
        br.forwardChain();
        resultado=TipoVehiculo.getValue();
        return resultado;
    }
    
    public String getVehiculo()
    {
        BaseReglas();
        TipoVehiculo.setValue("Aereo");
        Num_Ruedas.setValue(Administrador.numr);
        TipoMotor.setValue(Administrador.motorTipo);
        br.forwardChain();
        resultado=Vehiculo.getValue();
        return resultado;
    }
    public void BaseReglas()
    {
        TipoVehiculo = new RuleVariable(br, "TipoVehiculo");
        Vehiculo = new RuleVariable(br, "Vehiculo");
        Num_Ruedas = new RuleVariable(br, "Num_Ruedas");
        Motor = new RuleVariable(br, "Motor");
        Volar = new RuleVariable(br, "Volar");
        TipoMotor = new RuleVariable(br, "TipoMotor");

        Condition igual = new Condition("=");
        Condition mayor = new Condition(">");

        Rule Aereo = new Rule(br, "Aereo", 
                new Clause[]{new Clause(Motor, igual, "si"),
            new Clause(Volar, igual, "si")
        }, new Clause(TipoVehiculo, igual, "Aereo"));

        Rule Avioneta = new Rule(br, "Avioneta", new Clause[]{
            new Clause(TipoVehiculo, igual, "Aereo"), new Clause(Num_Ruedas, igual, "2"),
            new Clause(TipoMotor, igual, "Elice")
        }, new Clause(Vehiculo, igual, "Avioneta"));

        Rule AvionComercial = new Rule(br, "AvionComercial", new Clause[]{
            new Clause(TipoVehiculo, igual, "Aereo"), new Clause(Num_Ruedas, 
                    igual, "8"), new Clause(TipoMotor, igual, "Turbina")
        },
                new Clause(Vehiculo, igual, "Avion Comercial"));

        Rule Helicoptero = new Rule(br, "Heli", new Clause[]{
            new Clause(TipoVehiculo, igual, "Aereo"),
            new Clause(TipoMotor, igual, "Rotor")
        }, new Clause(Vehiculo, igual, "Helicoptero"));
    }
}

class ListoE extends OneShotBehaviour
{

    @Override
    public void action()
    {
        ACLMessage msg = new ACLMessage(ACLMessage.INFORM);
        msg.addReceiver(new AID("Coordinador", AID.ISLOCALNAME));
        msg.setContent("Listo");
        ((AereoAgent) myAgent).send(msg);
    }
}

class TipoVE extends CyclicBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.REQUEST);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Coordinador", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((AereoAgent) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Agente Aereo> Recibe peticion de Tipo de vehiculo de Coordinado");
            if (msg4.getContent().equals("TipoVehiculo"))
            {
                String conte = ((AereoAgent) myAgent).getTipoV();
                ACLMessage msg5 = new ACLMessage(ACLMessage.PROPOSE);
                msg5.addReceiver(new AID("Coordinador", AID.ISLOCALNAME));
                msg5.setContent(conte);
                ((AereoAgent) myAgent).send(msg5);
                (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Agente Aereo> Envia propuesta de Tipo de vehiculo a Coordinador");
            }
        }
    }
}

class getVehiculoE extends CyclicBehaviour
{
    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.REQUEST_WHEN);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Coordinador", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((AereoAgent) myAgent).blockingReceive(tem3);
        if(msg4!=null)
        {
            ACLMessage msg = new ACLMessage(ACLMessage.CONFIRM);
            msg.addReceiver(new AID("Coordinador",AID.ISLOCALNAME));            
            String vehiculo=((AereoAgent) myAgent).getVehiculo();
            msg.setContent(vehiculo);
            System.out.println("Vehiculo= "+vehiculo);
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText()+
                    "\n<Agente Aereo> Envia confirmacion de vehiculo= "+vehiculo+" a Coordinador");
            ((AereoAgent) myAgent).send(msg);
        }
    }    
}
