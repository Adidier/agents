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
public class AutoAgent extends Agent{
     BooleanRuleBase br = new BooleanRuleBase("test");
//    declaracion de cariable de reglas
    RuleVariable TipoVehiculo;
    RuleVariable Num_Ruedas;
    RuleVariable Motor;
    RuleVariable Vehiculo;
    RuleVariable Size;
    RuleVariable Num_Puertas;
    String resultado;
    
  
    @Override
    public void setup()
    {
        addBehaviour(new ListoA());
        addBehaviour(new TipoVA());
        addBehaviour(new getVehiculoA());
    }
    
    public String getTipoV()
    {
        BaseReglas();
        Num_Ruedas.setValue((Administrador.numr));
        Motor.setValue((Administrador.motor));
        br.forwardChain();
        resultado=TipoVehiculo.getValue();
        System.out.println("Resultado "+resultado);
        return resultado;        
    }
    
    public String getVehiculo()
    {
        BaseReglas();
        Num_Ruedas.setValue((Administrador.numr));
        Motor.setValue((Administrador.motor));
        TipoVehiculo.setValue((Administrador.tipo_V));
        Size.setValue((Administrador.size));
        Num_Puertas.setValue((Administrador.num_P));
        br.forwardChain();
        resultado=Vehiculo.getValue();
        return resultado;
    }
    
    public void BaseReglas()
    {
//        instancias de variables de regla
        TipoVehiculo = new RuleVariable(br, "TipoVehiculo");
        Num_Ruedas = new RuleVariable(br, "Num_Ruedas");
        Motor = new RuleVariable(br, "Num_Ruedas");
        Vehiculo = new RuleVariable(br, "Vehiculo");
        Size = new RuleVariable(br, "Size");
        Num_Puertas = new RuleVariable(br, "Num_Puertas");


//        definicion de operadores logicos (condicionales)
        Condition igual = new Condition("=");
        Condition diferente = new Condition("!=");
        Condition menor = new Condition("<");
        
         Rule Automobile = new Rule(br, "Automobile", new Clause[]{new Clause(Num_Ruedas, igual, "4"),
            new Clause(Motor, igual, "yes")
        }, new Clause(TipoVehiculo, igual, "automobile"));
         
        Rule SportsCar = new Rule(br, "sportsCar", 
                new Clause[]{new Clause(TipoVehiculo, igual, "automobile"),
            new Clause(Size, igual, "Pequeño"),
            new Clause(Num_Puertas, igual, "2")
        }, new Clause(Vehiculo, igual, "Sports_Car"));

        Rule Sedan = new Rule(br, "sedan", new Clause[]{new Clause(TipoVehiculo, igual, "automobile"),
            new Clause(Size, igual, "Mediano"),
            new Clause(Num_Puertas, igual, "4")
        }, new Clause(Vehiculo, igual, "Sedan"));

        Rule MiniVan = new Rule(br, "miniVan", new Clause[]{new Clause(TipoVehiculo, igual, "automobile"),
            new Clause(Size, igual, "Mediano"),
            new Clause(Num_Puertas, igual, "3")
        }, new Clause(Vehiculo, igual, "MiniVan"));

        Rule SUV = new Rule(br, "SUV", new Clause[]{new Clause(TipoVehiculo, igual, "automobile"),
            new Clause(Size, igual, "Grande"),
            new Clause(Num_Puertas, igual, "4")
        }, new Clause(Vehiculo, igual, "Sports_Utility_Vehicle"));
    }
}

class ListoA extends OneShotBehaviour
{

    @Override
    public void action()
    {
        ACLMessage msg=new ACLMessage(ACLMessage.INFORM);
        msg.addReceiver(new AID("Coordinador",AID.ISLOCALNAME));
        msg.setContent("Listo");
        ((AutoAgent)myAgent).send(msg);
    }
    
}

class TipoVA extends CyclicBehaviour
{

    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.REQUEST);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Coordinador", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((AutoAgent) myAgent).blockingReceive(tem3);
        if (msg4 != null)
        {
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText()+"\n<Agente Auto> Recibe peticion de Tipo de vehiculo a Coordinador");
            if (msg4.getContent().equals("TipoVehiculo"))
            {
                String conte=((AutoAgent) myAgent).getTipoV();
                ACLMessage msg5 = new ACLMessage(ACLMessage.PROPOSE);
                msg5.addReceiver(new AID("Coordinador", AID.ISLOCALNAME));
                msg5.setContent(conte);
                ((AutoAgent) myAgent).send(msg5);
                (Principal.AgentMessage).setText((Principal.AgentMessage).getText()+"\n<Agente Auto> Envia propuesta de Tipo de vehiculo a Coordinador");
            }
        }
        
    }
}

class getVehiculoA extends CyclicBehaviour
{
    @Override
    public void action()
    {
        MessageTemplate tem1 = MessageTemplate.MatchPerformative(ACLMessage.REQUEST_WHEN);
        MessageTemplate tem2 = MessageTemplate.MatchSender(new AID("Coordinador", AID.ISLOCALNAME));
        MessageTemplate tem3 = MessageTemplate.and(tem1, tem2);
        ACLMessage msg4 = ((AutoAgent) myAgent).blockingReceive(tem3);
        if(msg4!=null)
        {
            ACLMessage msg = new ACLMessage(ACLMessage.CONFIRM);
            msg.addReceiver(new AID("Coordinador",AID.ISLOCALNAME));
            
            String vehiculo=((AutoAgent) myAgent).getVehiculo();
            msg.setContent(vehiculo);
            System.out.println("Vehiculo= "+vehiculo);
            (Principal.AgentMessage).setText((Principal.AgentMessage).getText()+
                    "\n<Agente Auto> Envia confirmacion de vehiculo= "+vehiculo+" a Coordinador");
            ((AutoAgent) myAgent).send(msg);
        }
    }    
}
