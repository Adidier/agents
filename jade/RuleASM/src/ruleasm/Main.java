/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ruleasm;

import jade.wrapper.AgentController;

/**
 *
 * @author Virgilio
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        Iniciar ag=new Iniciar("localhost",null);
        AgentController ag1= ag.iniciaAgente("Coordinador", "ruleasm.Administrador");
        AgentController ag2= ag.iniciaAgente(null,null,"Cyclic", "ruleasm.CyclicAgent");
        AgentController ag3= ag.iniciaAgente(null,null,"Auto", "ruleasm.AutoAgent");
        AgentController ag4= ag.iniciaAgente(null, null, "Aereo", "ruleasm.AereoAgent");   
    }
}
