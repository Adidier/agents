/*
 * Iniciar.java
 *
 * Created on 14 de febrero de 2025, 01:17 PM
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ruleasm;
import jade.core.Agent;
import jade.core.Profile;
import jade.lang.acl.*;
// import jade.lang.*;
import jade.wrapper.AgentController;
import jade.wrapper.ContainerController;
import jade.domain.FIPAAgentManagement.*;
import jade.core.*;
import jade.core.Runtime;

/**
 *
 * @author Virgilio
 */
public class Iniciar extends Agent
{
    Runtime rt =Runtime.instance();
    AgentController ac;
    ContainerController cc;
    /** Creates a new instance of Iniciar */
    public Iniciar(String host, String port)
    {
        Profile p= new ProfileImpl();
        p.setParameter(Profile.MAIN_HOST,host);
        p.setParameter(Profile.MAIN_PORT,port);
        cc= rt.createMainContainer(p);
    }
    public AgentController iniciaAgente(String name, String nameClass)
    {
        if(cc!=null)
        {
            try
            {
                ac=cc.createNewAgent(name,nameClass,null);
                ac.start();
                return ac;
            }
            catch(Exception e)
            {
                e.printStackTrace();
            }
        }
        return null;
    }
    public AgentController iniciaAgente(String host, String port, String name, String nameClass)
    {
        if(cc!=null)
        {
            try
            {
                ac=cc.createNewAgent(name,nameClass,null);
                ac.start();
                return ac;
            }
            catch(Exception e)
            {
                e.printStackTrace();
            }
        }
        else
        {
            System.out.println("-----> Contenedor vacio");
        }
        return null;
    }
}


