/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ruleasm;

import jade.core.Agent;
import Rule.*;
import jade.core.AID;
import jade.core.behaviours.*;
import jade.domain.FIPAAgentManagement.*;
import jade.lang.acl.*;
import jade.lang.acl.ACLMessage;
import jade.domain.FIPAAgentManagement.*;
import jade.lang.acl.MessageTemplate;

/**
 *
 * @author Virgilio
 */
public class CyclicAgent extends Agent
{

    BooleanRuleBase br = new BooleanRuleBase("test");
//    declaracion de cariable de reglas
    RuleVariable TipoVehiculo;
    RuleVariable Num_Ruedas;
    RuleVariable Motor;
    RuleVariable Vehiculo;
    String resultado;

    @Override
    public void setup()
    {
        addBehaviour(new ListoC());
        addBehaviour(new TipoVC());
        addBehaviour(new getVehiculoC());
    }

    public String getTipoV()
    {
        BaseReglas();
        Num_Ruedas.setValue((Administrador.numr));
        System.out.println("Numero de Ruedas " + (Administrador.numr));
        Motor.setValue((Administrador.motor));
        br.forwardChain();
        resultado = TipoVehiculo.getValue();
        System.out.println("Resultado " + resultado);
        return resultado;
    }
    
    public String getVehiculo()
    {
        BaseReglas();
        Num_Ruedas.setValue((Administrador.numr));
        System.out.println("Numero de Ruedas " + (Administrador.numr));
        Motor.setValue((Administrador.motor));
        System.out.println("Motor " + (Administrador.motor));
        TipoVehiculo.setValue("cycle");
        br.forwardChain();
        resultado = Vehiculo.getValue();
        System.out.println("Resultado " + resultado);
        return resultado;
    }

    public void BaseReglas()
    {
        TipoVehiculo = new RuleVariable(br, "TipoVehiculo");
        Num_Ruedas = new RuleVariable(br, "Num_Ruedas");
        Motor = new RuleVariable(br, "Num_Ruedas");
        Vehiculo = new RuleVariable(br, "Vehiculo");
        
        Condition igual = new Condition("=");
        Condition diferente = new Condition("!=");
        Condition menor = new Condition("<");

        Rule Cycle = new Rule(br, "Cycle", new Clause(Num_Ruedas, menor, "4"), new Clause(TipoVehiculo, igual, "cycle"));

        Rule Bicycle = new Rule(br, "bicycle", new Clause[]{new Clause(TipoVehiculo, igual, "cycle"),
            new Clause(Num_Ruedas, igual, "2"),
            new Clause(Motor, igual, "no")
        }, new Clause(Vehiculo, igual, "Bicycle"));

        Rule Tricycle = new Rule(br, "tricycle", new Clause[]{new Clause(TipoVehiculo, igual, "cycle"),
            new Clause(Num_Ruedas, igual, "3"),
            new Clause(Motor, igual, "no")
        }, new Clause(Vehiculo, igual, "Tricycle"));

        Rule Motorcycle = new Rule(br, "motorcycle", new Clause[]{new Clause(TipoVehiculo, igual, "cycle"),
            new Clause(Num_Ruedas, igual, "2"),
            new Clause(Motor, igual, "yes")
        }, new Clause(Vehiculo, igual, "Motorcycle"));
    }
}

class ListoC extends OneShotBehaviour
{

    @Override
    public void action()
    {
        ACLMessage msg = new ACLMessage(ACLMessage.INFORM);
        msg.addReceiver(new AID("Coordinador", AID.ISLOCALNAME));
        msg.setContent("Listo");
        ((CyclicAgent) myAgent).send(msg);
    }
}

class TipoVC extends CyclicBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.REQUEST);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Coordinador", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = myAgent.blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Agente Cyclic> Recibe peticion de Tipo de vehiculo de Coordinado");
            if (msg4.getContent().equals("TipoVehiculo"))
            {
                String conte = ((CyclicAgent) myAgent).getTipoV();
                ACLMessage msg5 = new ACLMessage(ACLMessage.PROPOSE);
                msg5.addReceiver(new AID("Coordinador", AID.ISLOCALNAME));
                msg5.setContent(conte);
                ((CyclicAgent) myAgent).send(msg5);
                (Principal.AgentMessage).setText((Principal.AgentMessage).getText() + "\n<Agente Cyclic> Envia propuesta de Tipo de vehiculo a Coordinador");
            }
        }
    }
}

class getVehiculoC extends CyclicBehaviour
{
    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.REQUEST_WHEN);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Coordinador", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((CyclicAgent) myAgent).blockingReceive(tem3);
        if(msg4!=null)
        {
            ACLMessage msg = new ACLMessage(ACLMessage.CONFIRM);
            msg.addReceiver(new AID("Coordinador",AID.ISLOCALNAME));            
            String vehiculo=((CyclicAgent) myAgent).getVehiculo();
            msg.setContent(vehiculo);
            System.out.println("Vehiculo= "+vehiculo);
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText()+
                    "\n<Agente Cyclic> Envia confirmacion de vehiculo= "+vehiculo+" a Coordinador");
            ((CyclicAgent) myAgent).send(msg);
        }
    }    
}

