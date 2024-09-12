#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <tuple>
#include <list>
#include <cstring>
#include <cstdlib>
using namespace std;

/*
Examples of configuration settings from:
https://github.com/CodeReclaimers/neat-python/blob/master/examples/circuits/config

[NEAT]
fitness_criterion     = max     # unknown
fitness_threshold     = -0.01   # unknown
pop_size              = 500     # population size
reset_on_extinction   = False   # if True, reset all species if all species are extinct

[CircuitGenome]
# component type options
component_default      = resistor           # unknown
component_mutate_rate  = 0.1                # unknown
component_options      = resistor diode     # unknown

# component value options
value_init_mean          = 4.5      # unknown
value_init_stdev         = 0.5      # unknown
value_max_value          = 6.0      # unknown
value_min_value          = 3.0      # unknown
value_mutate_power       = 0.1      # unknown
value_mutate_rate        = 0.8      # unknown
value_replace_rate       = 0.1      # unknown

# genome compatibility options
compatibility_disjoint_coefficient = 1.0    # unknown
compatibility_weight_coefficient   = 1.0    # unknown

# connection add/remove rates
conn_add_prob           = 0.2   # probability of adding a connection
conn_delete_prob        = 0.2   # probability of deleting a connection

# connection enable options
enabled_default         = True  # probability of a connection being enabled/disabled
enabled_mutate_rate     = 0.02  # probability of a connection being enabled/disabled

# node add/remove rates
node_add_prob           = 0.1   # probability of adding a node
node_delete_prob        = 0.1   # probability of deleting a node

# network parameters
num_inputs              = 3  # number of input nodes
num_outputs             = 1 # number of output nodes

[DefaultSpeciesSet]
compatibility_threshold = 2.0   # some threshold for compatibility between species, not sure if I'll use this

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
*/

// make a variable type for the data type

float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") {};
};

class Node
{
private:
    int id;      // Unique identifier for the node in the genome
    float value; // Value of the node
    // float bias;                                    // TODO: implement bias
    list<tuple<Node *, float, bool>> *connections; // tuple<from_node, weight, enabled>
    bool ready;                                    // Flag to indicate if the node has been calculated in the current feed forward pass
    bool connection_enabled;                       // Flag to indicate if any of the connections to the node are enabled

    bool CheckConnectionsEnabled(); // Check if the node has any active connections

public:
    // Constructors
    Node(int id, float value); // Constructor
    ~Node();                   // Destructor

    // Methods
    void CalculateValue();                                         // Recursively calculate the value of the node
    void AddConnection(Node *node, float weight, bool enabled);    // Add a connection to the node
    void updateConnection(Node *node, float weight, bool enabled); // Update a connection to the node
    void LoadValue(float value);                                   // Load a value into the node, used for input nodes
    void Reset();                                                  // Reset the node to not ready

    // Getters
    int GetId() { return id; }                                                  // Get the id of the node
    float GetValue() { return value; }                                          // Get the value of the node
    bool GetConnectionsEnabled() { return connection_enabled; }                 // Check if the node has any active connections
    list<tuple<Node *, float, bool>> *GetConnections() { return connections; }; // Get the connections of the node
};

Node *GetNodeWithId(Node **nodes, int numNodes, int id)
{
    /**
     * Helper function to get a node with a specific id from an array of nodes.
     *
     * @param nodes: Array of nodes
     * @param id: Id of the node to get
     *
     * @return: Node with the specified id, or NULL if the node is not found
     */
    for (int i = 0; i < numNodes; i++)
    {
        if (nodes[i]->GetId() == id)
        {
            return nodes[i];
        }
    }
    return NULL;
}

Node::Node(int id, float value)
{
    /**
     * Constructor for the Node class.
     *
     * @param id: Unique identifier for the node in the genome
     * @param value: Value of the node
     */

    this->id = id;                                        // Set the id of the node
    this->value = value;                                  // Set the value of the node
    ready = false;                                        // Set the ready flag to false
    connections = new list<tuple<Node *, float, bool>>(); // Initialize the connections list
    connection_enabled = false;                           // Set the connections_disabled flag to true
}

Node::~Node()
{
    /**
     * Destructor for the Node class.
     */
    delete connections;
}

void Node::AddConnection(Node *node, float weight, bool enabled)
{
    /**
     * Add a connection to the node.
     *
     * @param node: Node to connect to
     * @param weight: Weight of the connection
     * @param enabled: Flag to indicate if the connection is enabled
     */
    connections->push_back(make_tuple(node, weight, enabled));
    if (!connection_enabled && enabled)
    {
        connection_enabled = true;
    }
}

void Node::updateConnection(Node *node, float weight, bool enabled)
{
    /**
     * Update a connection to the node.
     *
     * @param node: Node to connect to
     * @param weight: Weight of the connection
     * @param enabled: Flag to indicate if the connection is enabled
     */

    bool found = false;
    for (tuple<Node *, float, bool> conn : *connections)
    {
        if (get<0>(conn) == node)
        {
            // tuples are immutable
            connections->remove(conn);
            found = true;
            break;
        }
    }
    if (!found)
    {
        cout << "Could not find connection to update" << endl;
        throw "Could not find connection to update";
    }
    connections->push_back(make_tuple(node, weight, enabled));
    if (!connection_enabled && enabled)
    {
        connection_enabled = true;
    }
    else if (connection_enabled && !enabled)
    {
        connection_enabled = CheckConnectionsEnabled();
    }
}

void Node::LoadValue(float value)
{
    /**
     * Load a value into the node, used for input nodes.
     *
     * @param value: Value to load into the node
     */
    this->value = value / 255; // Normalize the value to be between 0 and 1 by dividing by 255
    ready = true;
}

void Node::CalculateValue()
{
    /**
     * Recursively calculate the value of the node.
     *
     * The value of the node is calculated by summing the values of the incoming nodes multiplied by the weights of the connections.
     * If the incoming node is not ready, the value of the incoming node is calculated by recursively calling CalculateValue.
     */

    // Check if node has any incoming connections, if not, set ready to true and do not update value
    if (connections->empty())
    {
        ready = true;
        return;
    }

    // Check if all incoming connections are disabled, if so, set ready to true and do not update value
    bool all_connections_disabled = true;
    for (tuple<Node *, float, bool> conn : *connections)
    {
        if (get<2>(conn))
        {
            all_connections_disabled = false;
            break;
        }
    }

    // If all incoming connections are disabled, set ready to true and do not update value
    if (all_connections_disabled)
    {
        ready = true;
        return;
    }

    // Calculate value recursively
    value = 0;
    ready = true;
    for (tuple<Node *, float, bool> conn : *connections)
    {
        if (get<2>(conn))
        {
            if (!get<0>(conn)->ready)
            {
                get<0>(conn)->CalculateValue();
            }
            value += get<0>(conn)->value * get<1>(conn);
        }
    }
    value = sigmoid(value);
}

void Node::Reset()
{
    /**
     * Reset the node to not ready, used to reset the state of the node between feed forward passes.
     */
    ready = false;
}

bool Node::CheckConnectionsEnabled()
{
    /**
     * Check if the node has any active connections.
     *
     * @return: True if the node has any active connections, False otherwise
     */
    for (tuple<Node *, float, bool> conn : *connections)
    {
        if (get<2>(conn))
        {
            return true;
        }
    }
    return false;
}

class Genome
{
private:
    string *name;   // Name of the genome
    float fitness;  // Fitness of the genome
    int numNodes;   // Number of nodes in the genome
    int numInputs;  // Number of input nodes
    int numOutputs; // Number of output nodes
    int numHidden;  // Number of hidden nodes

    Node **nodes;   // Array of nodes in the genome
    Node *bias;     // Bias node
    Node **inputs;  // Array of input nodes
    Node **outputs; // Array of output nodes
    Node **hidden;  // Array of hidden nodes

    random_device rd;                                          // Random device
    default_random_engine generator;                           // Random number generator
    uniform_real_distribution<float> neg_uniform_distribution; // Normal distribution with mean -1 and standard deviation 1
    uniform_real_distribution<float> pos_uniform_distribution; // Normal distribution with mean 0 and standard deviation 1

public:
    // Constructors/Destructors
    Genome(string *name);  // Constructor
    ~Genome();             // Destructor
    Genome(Genome *other); // Copy constructor

    // Methods
    void InitGenome(int numInputs, int numOutputs); // Initialize a genome with a given number of input and output nodes and connections from each output node to the bias node
    void Load(string *filename);                    // Load a genome from a file
    void Save(string *filename);                    // Save a genome to a file
    Genome *Crossover(Genome *other);               // Crossover the genome with another genome
    int FeedForward(float *input_image);            // Feed forward the input image through the genome and return the index of the output node with the highest value

    // Mutate methods
    void Mutate();                         // Mutate the genome
    void MutateAdjustConnectionWeight();   // Adjust the weight of a connection in the genome
    void MutateAddConnection();            // Add a new connection to the genome
    void MutateAddNode();                  // Add a new node to the genome
    void MutateEnableConnection();         // Enable a connection in the genome
    void MutateDisableConnection();        // Disable a connection in the genome
    void MutateChangeActivationFunction(); // Change the activation function of a node in the genome

    // Getters/Setters
    void SetName(string *name) { this->name = name; }           // Set the name of the genome
    string *GetName() { return name; }                          // Get the name of the genome
    int GetNumNodes() { return numNodes; }                      // Get the number of nodes in the genome
    int GetNumInputs() { return numInputs; }                    // Get the number of input nodes in the genome
    int GetNumOutputs() { return numOutputs; }                  // Get the number of output nodes in the genome
    int GetNumHidden() { return numHidden; }                    // Get the number of hidden nodes in the genome
    Node *GetNode(int index) { return nodes[index]; }           // Get a node in the genome
    void SetFitness(float fitness) { this->fitness = fitness; } // Set the fitness of the genome
    float GetFitness() { return fitness; }                      // Get the fitness of the genome
    void PrintInfo();                                           // Print the information of the genome

    // TODO: move this out of the class and make it a friend function
    Node *FindRandomNodeWithEnabledConnection(); // Find a random node with an enabled connection

    void PrintOutputLayerValues();
};

bool doesGenomeHaveConnections(Genome *genome);

Genome::Genome(string *name)
{
    /**
     * Constructor for the Genome class.
     *
     * @param name: Name of the genome
     */
    this->name = new string(name->c_str());                                 // Set the name of the genome
    this->numInputs = 0;                                                    // Set the number of input nodes
    this->numOutputs = 0;                                                   // Set the number of output nodes
    this->numHidden = 0;                                                    // Set the number of hidden nodes
    generator = default_random_engine(rd());                                // Initialize the random number generator
    neg_uniform_distribution = uniform_real_distribution<float>(-1.0, 1.0); // Initialize the negative normal distribution
    pos_uniform_distribution = uniform_real_distribution<float>(0.0, 1.0);  // Initialize the positive normal distribution
}

Genome::~Genome()
{
    /**
     * Destructor for the Genome class.
     */
    // delete[] name;
    // delete[] nodes;
    // delete[] inputs;
    // delete[] outputs;
    // delete[] hidden;
}

Genome::Genome(Genome *other)
{
    /**
     * Copy constructor for the Genome class.
     *
     * @param other: Genome to copy
     */
    this->name = new string(other->name->c_str());
    this->fitness = 0;
    this->numNodes = other->numNodes;
    this->numInputs = other->numInputs;
    this->numOutputs = other->numOutputs;
    this->numHidden = other->numHidden;

    this->nodes = new Node *[numNodes];
    for (int i = 0; i < numNodes; i++)
    {
        this->nodes[i] = new Node(other->nodes[i]->GetId(), other->nodes[i]->GetValue());
    }
    this->bias = this->nodes[0];
    this->inputs = new Node *[numInputs];
    for (int i = 0; i < numInputs; i++)
    {
        int id = other->inputs[i]->GetId();
        inputs[i] = GetNodeWithId(nodes, numNodes, id);
    }
    this->outputs = new Node *[numOutputs];
    for (int i = 0; i < numOutputs; i++)
    {
        int id = other->outputs[i]->GetId();
        outputs[i] = GetNodeWithId(nodes, numNodes, id);
    }
    this->hidden = new Node *[numHidden];
    for (int i = 0; i < numHidden; i++)
    {
        int id = other->hidden[i]->GetId();
        hidden[i] = GetNodeWithId(nodes, numNodes, id);
    }

    // Copy connections
    // for every node in the other genome, find the corresponding node in this genome and copy the connections
    for (int i = 0; i < numNodes; i++)
    {
        Node *current_node = GetNodeWithId(other->nodes, numNodes, i);
        if (current_node == NULL)
        {
            cout << "NULL Could not find node with id " << i << endl;
            throw "Could not find node with id";
        }
        list<tuple<Node *, float, bool>> *connections = current_node->GetConnections();
        for (tuple<Node *, float, bool> conn : *connections)
        {
            int from_id = get<0>(conn)->GetId();
            Node *from_node = GetNodeWithId(other->nodes, numNodes, from_id);
            float weight = get<1>(conn);
            bool enabled = get<2>(conn);
            this->nodes[i]->AddConnection(from_node, weight, enabled);
        }
    }

    neg_uniform_distribution = uniform_real_distribution<float>(-1.0, 1.0);
    pos_uniform_distribution = uniform_real_distribution<float>(0.0, 1.0);
}

void Genome::InitGenome(int numInputs, int numOutputs)
{
    /**
     * Initialize a genome with a given number of input and output nodes and connections from each output node to the bias node.
     *
     * @param numInputs: Number of input nodes
     * @param numOutputs: Number of output nodes
     */

    // Set the number of input and output nodes
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;

    // Calculate the number of nodes
    numNodes = numInputs + numOutputs + 1;

    // Initialize the nodes array
    nodes = new Node *[numNodes];

    // Initialize the bias node
    bias = new Node(0, 1);
    nodes[0] = bias;

    // Initialize the rest of the nodes
    for (int i = 0; i < numNodes - 1; i++)
    {
        nodes[i + 1] = new Node(i + 1, -INFINITY);
    }

    // Set the inputs
    inputs = new Node *[numInputs];
    for (int i = 0; i < numInputs; i++)
    {
        inputs[i] = nodes[i + 1];
    }

    // Set the outputs
    outputs = new Node *[numOutputs];
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i] = nodes[numInputs + i + 1];
    }

    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i]->AddConnection(bias, neg_uniform_distribution(generator), true);
    }
}

void Genome::Load(string *filename)
{
    /**
     * Load a genome from a file.
     *
     * The file format is as follows:
     *  - First line: numInputs, numHidden, numOutputs
     *  - Second line: input node ids
     *  - Third line: hidden node ids
     *  - Fourth line: output node ids
     *  - For each node:
     *   - id, value, num_connections, conn0_from_id, conn0_weight, conn0_enabled, conn1_from_id, conn1_weight, conn1_enabled, ...
     *
     * @param filename: Name of the file to load the genome from
     */

    ifstream MyFile(*filename);

    // first line: numInputs, numHidden, numOutputs
    MyFile >> numInputs >> numHidden >> numOutputs;

    numNodes = numInputs + numHidden + numOutputs + 1;

    // second line: input node ids
    int *input_ids = new int[numInputs];
    for (int i = 0; i < numInputs; i++)
    {
        MyFile >> input_ids[i];
    }

    // third line: hidden node ids
    int *hidden_ids = new int[numHidden];
    for (int i = 0; i < numHidden; i++)
    {
        MyFile >> hidden_ids[i];
    }

    // fourth line: output node ids
    int *output_ids = new int[numOutputs];
    for (int i = 0; i < numOutputs; i++)
    {
        MyFile >> output_ids[i];
    }

    // for each node: load node
    nodes = new Node *[numNodes];
    list<tuple<int, int, float, bool>> *connections = new list<tuple<int, int, float, bool>>;
    for (int i = 0; i < numNodes; i++)
    {
        int id;
        float value;
        int num_connections;
        MyFile >> id >> value >> num_connections;

        // Create the node
        nodes[i] = new Node(id, value);

        for (int j = 0; j < num_connections; j++)
        {
            int from_id;
            float weight;
            bool enabled;
            MyFile >> from_id >> weight >> enabled;
            connections->push_back(make_tuple(id, from_id, weight, enabled));
        }
    }
    MyFile.close();

    for (tuple<int, int, float, bool> conn : *connections)
    {
        int to_id = get<0>(conn);
        int from_id = get<1>(conn);
        float weight = get<2>(conn);
        bool enabled = get<3>(conn);
        Node *to_node = GetNodeWithId(nodes, numNodes, to_id);
        Node *from_node = GetNodeWithId(nodes, numNodes, from_id);
        to_node->AddConnection(from_node, weight, enabled);
    }

    // Set the inputs, outputs, and hidden nodes
    inputs = new Node *[numInputs];
    for (int i = 0; i < numInputs; i++)
    {
        inputs[i] = GetNodeWithId(nodes, numNodes, input_ids[i]);
        if (inputs[i] == NULL)
        {
            cout << "ACould not find input node with id " << input_ids[i] << endl;
            throw "Could not find input node with id";
        }
    }

    bias = nodes[0];

    outputs = new Node *[numOutputs];
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i] = GetNodeWithId(nodes, numNodes, output_ids[i]);
        if (outputs[i] == NULL)
        {
            cout << "BCould not find output node with id " << output_ids[i] << endl;
            throw "Could not find output node with id";
        }
    }

    hidden = new Node *[numHidden];
    for (int i = 0; i < numHidden; i++)
    {
        hidden[i] = GetNodeWithId(nodes, numNodes, hidden_ids[i]);
        if (hidden[i] == NULL)
        {
            cout << "CCould not find hidden node with id " << hidden_ids[i] << endl;
            throw "Could not find hidden node with id";
        }
    }
}

void Genome::Save(string *filename)
{
    /**
     * Save a genome to a file.
     *
     * The file format is as follows:
     * - First line: numInputs, numHidden, numOutputs
     * - Second line: input node ids
     * - Third line: hidden node ids
     * - Fourth line: output node ids
     * - For each node:
     * - id, value, num_connections, from_id0, weight0, enabled0, from_id1, weight1, enabled1, ...
     *
     * @param filename: Name of the file to save the genome to
     */

    // Create file
    ofstream MyFile(*filename);

    // first line: numInputs, numHidden, numOutputs
    MyFile << numInputs << " " << numHidden << " " << numOutputs << endl;

    // second line: input node ids
    for (int i = 0; i < numInputs; i++)
    {
        MyFile << inputs[i]->GetId() << " ";
    }
    MyFile << endl;

    // third line: hidden node ids
    for (int i = 0; i < numHidden; i++)
    {
        MyFile << hidden[i]->GetId() << " ";
    }
    MyFile << endl;

    // fourth line: output node ids
    for (int i = 0; i < numOutputs; i++)
    {
        MyFile << outputs[i]->GetId() << " ";
    }
    MyFile << endl;

    // for each node save node as:
    // id, value, num_connections, from_id0, weight0, enabled0, from_id1, weight1, enabled1, ...
    for (int i = 0; i < numNodes; i++)
    {
        list<tuple<Node *, float, bool>> *conns = nodes[i]->GetConnections();
        MyFile << nodes[i]->GetId() << " ";
        if (nodes[i]->GetValue() == -INFINITY)
        {
            MyFile << 1.17549e-38;
        }
        else if (nodes[i]->GetValue() == NAN)
        {
            cout << "Saving Node " << i << " but the value is NAN" << endl;
            throw "Saving Node but the value is NAN";
        }
        else
        {
            MyFile << nodes[i]->GetValue();
        }
        MyFile << " " << conns->size() << " ";

        for (tuple<Node *, float, bool> conn : *conns)
        {
            MyFile << get<0>(conn)->GetId() << " " << get<1>(conn) << " " << get<2>(conn) << " ";
        }
        MyFile << endl;
    }

    MyFile.close();
}

Node *Genome::FindRandomNodeWithEnabledConnection()
{
    if (numNodes == 0)
    {
        cout << "No nodes in genome, cannot FindRandomNodeWithEnabledConnection" << endl;
        return NULL;
    }
    // TODO: WARN possible infinite loop, need to make sure there is always at least one enabled connection in the genome
    int num_tries = 0;
    while (true)
    {
        if (num_tries > numNodes * 2)
        {
            cout << "Could not find node with enabled connection" << endl;
            return NULL;
        }

        int random_node_index = rand() % (numHidden + numOutputs);
        // cout << "random_node_index: " << random_node_index << endl;
        Node *random_node;
        if (random_node_index < numHidden)
        {
            random_node = hidden[random_node_index];
        }
        else
        {
            random_node = outputs[random_node_index - numHidden];
        }
        // cout << "random_node: " << random_node->GetId() << endl;
        list<tuple<Node *, float, bool>> *connections = random_node->GetConnections();
        // cout << "conns size: " << connections->size() << endl;
        for (tuple<Node *, float, bool> conn : *connections)
        {
            if (get<2>(conn))
            {
                return random_node;
            }
        }

        num_tries++;
    }
}

void Genome::Mutate()
{
    /**
     * Mutate the genome.
     *
     * The mutation default rates are as follows:
     * - mutation 1: adjust the weight of a connection (50%)
     * - mutation 2: add a new connection (10%)
     * - mutation 3: add a new node (10%)
     * - mutation 4: disable a connection (10%)
     * - mutation 5: enable a connection (10%)
     * - mutation 6: change an activation function (10%) DISABLED
     */

    // cout << "Genome mutating" << endl;

    // float mutation = pos_uniform_distribution(generator);
    int mutation = rand() % 9;
    // cout << "Mutation: " << mutation << endl;
    if (mutation < 5)
    {
        // mutation 1: adjust the weight of a connection
        MutateAdjustConnectionWeight();
    }
    else if (mutation < 6)
    {
        try
        {
            // mutation 2: add a new connection
            MutateAddConnection();
        }
        catch (const NotImplemented &e)
        {
            // cout << "\t\t^ not implemented, defaulting to adjusting connection" << endl;
            MutateAdjustConnectionWeight();
        }
    }
    else if (mutation < 7)
    {
        try
        {
            // mutation 3: add a new node
            MutateAddNode();
        }
        catch (const NotImplemented &e)
        {
            // cout << "\t\t^ not implemented, defaulting to adjusting connections" << endl;
            MutateAdjustConnectionWeight();
        }
    }
    else if (mutation < 8)
    {
        try
        {
            // mutation 4: disable a connection
            MutateDisableConnection();
        }
        catch (const NotImplemented &e)
        {
            // cout << "\t\t^ not implemented, defaulting to adjusting connection" << endl;
            MutateAdjustConnectionWeight();
        }
    }
    // else if (mutation < 9)
    else
    {
        try
        {
            // mutation 5: enable a connection
            MutateEnableConnection();
        }
        catch (const NotImplemented &e)
        {
            // cout << "\t\t^ not implemented, defaulting to adjusting connection" << endl;
            MutateAdjustConnectionWeight();
        }
    }
    // else
    // {
    //     try
    //     {
    //         // mutation 6: change an activation function
    //         MutateChangeActivationFunction();
    //     }
    //     catch (const NotImplemented &e)
    //     {
    //         cout << "\t\t^ not implemented, defaulting to adjusting connection" << endl;
    //         MutateAdjustConnectionWeight();
    //     }
    // }
}

void Genome::MutateAdjustConnectionWeight()
{
    // cout << "Genome mutation 1: adjust the weight of a connection" << endl;
    Node *random_node = FindRandomNodeWithEnabledConnection();
    if (random_node == NULL)
    {
        cout << "No enabled connections in genome, cannot mutate" << endl;
        return;
    }
    while (true)
    {
        list<tuple<Node *, float, bool>> *connections = random_node->GetConnections();
        int random_connection_index = rand() % connections->size();
        std::list<tuple<Node *, float, bool>>::iterator it = connections->begin();
        advance(it, random_connection_index);
        tuple<Node *, float, bool> random_connection = *it;
        connections->erase(it);
        Node *from_node = get<0>(random_connection);
        float weight = get<1>(random_connection);
        bool enabled = get<2>(random_connection);
        if (!enabled)
        {
            continue;
        }
        weight += pos_uniform_distribution(generator);
        // check that the weight is within the bounds
        if (weight > 1.0)
        {
            weight = 1.0;
        }
        else if (weight < -1.0)
        {
            weight = -1.0;
        }
        connections->push_back(make_tuple(from_node, weight, enabled));
        return;
    }
}

void Genome::MutateAddConnection()
{
    // cout << "Genome mutation 2: add a new connection" << endl;
    // first, get start node from input, hidden, or bias
    int random_node_index = rand() % (numInputs + numHidden + 1);
    Node *start_node;
    if (random_node_index < numInputs)
    {
        start_node = inputs[random_node_index];
    }
    else if (random_node_index < numInputs + numHidden)
    {
        start_node = hidden[random_node_index - numInputs];
    }
    else
    {
        start_node = bias;
    }

    // second, get end node from hidden or output
    random_node_index = rand() % (numHidden + numOutputs);
    Node *end_node;
    if (random_node_index < numHidden)
    {
        end_node = hidden[random_node_index];
    }
    else
    {
        end_node = outputs[random_node_index - numHidden];
    }

    // third, check that the connection does not already exist
    bool connection_exists = false;
    list<tuple<Node *, float, bool>> *connections = end_node->GetConnections();
    for (tuple<Node *, float, bool> conn : *connections)
    {
        if (get<0>(conn) == start_node)
        {
            connection_exists = true;
            break;
        }
    }
    if (connection_exists)
    {
        cout << "Connection already exists, cannot add new connection" << endl;
        return;
    }

    // fourth, check that the connection does not create a cycle
    // TODO: implement cycle detection

    // fifth, add the new connection
    end_node->AddConnection(start_node, neg_uniform_distribution(generator), true);
}

void Genome::MutateAddNode()
{
    // cout << "Genome mutation 3: add a new node" << endl;

    // first, get a random connection
    Node *random_node = FindRandomNodeWithEnabledConnection();
    list<tuple<Node *, float, bool>> *connections = random_node->GetConnections();
    int random_connection_index = rand() % connections->size();
    std::list<tuple<Node *, float, bool>>::iterator it = connections->begin();
    advance(it, random_connection_index);
    tuple<Node *, float, bool> random_connection = *it;
    Node *from_node = get<0>(random_connection);
    float weight = get<1>(random_connection);
    connections->erase(it);

    // second, disable the connection
    connections->push_back(make_tuple(from_node, weight, false));

    // third, create a new node

    // extend the nodes array
    Node **new_nodes = new Node *[numNodes + 1];
    for (int i = 0; i < numNodes; i++)
    {
        new_nodes[i] = nodes[i];
    }
    // add the new node
    new_nodes[numNodes] = new Node(numNodes, -INFINITY);
    delete[] nodes;
    nodes = new_nodes;

    // extend the hidden nodes array
    Node **new_hidden = new Node *[numHidden + 1];
    for (int i = 0; i < numHidden; i++)
    {
        new_hidden[i] = hidden[i];
    }
    new_hidden[numHidden] = nodes[numNodes];
    if (numHidden > 0)
    {
        delete[] hidden;
    }
    hidden = new_hidden;

    // fourth, create a new connection from the start node to the new node
    float new_weight = neg_uniform_distribution(generator);
    nodes[numNodes]->AddConnection(from_node, new_weight, true);

    // fifth, create a new connection from the new node to the end node
    new_weight = neg_uniform_distribution(generator);
    random_node->AddConnection(nodes[numNodes], new_weight, true);

    numNodes++;
    numHidden++;
}

void Genome::MutateEnableConnection()
{
    // TODO: implement this mutation

    // cout << "Genome mutation 5: enable a connection" << endl;

    // first, get a random connection

    // second, if the connection is already enabled, get a new connection

    // third, check that enabling the connection does not create a cycle

    // fourth, enable the connection

    throw NotImplemented();
}

void Genome::MutateDisableConnection()
{
    // TODO: implement this mutation

    // cout << "Genome mutation: disable a connection" << endl;

    // first, get a random connection

    // second, if the connection is already disabled, get a new connection

    // third, check that the connection is not the only active connection left in the genome

    // fourth, disable the connection

    // Node *random_node = FindRandomNodeWithEnabledConnection();
    // list<tuple<Node *, float, bool>> *connections = random_node->GetConnections();
    // int random_connection_index = rand() % connections->size();
    // std::list<tuple<Node *, float, bool>>::iterator it = connections->begin();
    // advance(it, random_connection_index);
    // tuple<Node *, float, bool> random_connection = *it;
    // Node *from_node = get<0>(random_connection);
    // float weight = get<1>(random_connection);
    // connections->erase(it);
    // connections->push_back(make_tuple(from_node, weight, false));

    throw NotImplemented();
}

void Genome::MutateChangeActivationFunction()
{
    // TODO: implement this mutation

    // cout << "Genome mutation 6: change an activation function" << endl;

    // first, get a random node from hidden nodes

    // second, choose a new activation function

    // third, check that the node does not already use the activation function

    // fourth, change the activation function

    throw NotImplemented();
}

Genome *Genome::Crossover(Genome *other)
{
    /**
     * Crossover this genome with another genome.
     *
     * TODO: implement crossover algorithm
     *
     * @param other: Genome to crossover with
     */

    Genome *child = new Genome(this->name);

    // get the shape of the new genome
    child->numInputs = numInputs;
    child->numOutputs = numOutputs;
    child->numHidden = max(numHidden, other->numHidden);
    child->numNodes = child->numInputs + child->numOutputs + child->numHidden + 1;

    // crossover process:
    // 1. find the matching genes(nodes) between the two genomes
    for (int i = 0; i < max(numNodes, other->numNodes); i++)
    {
        Node *a = GetNodeWithId(nodes, numNodes, i);
        Node *b = GetNodeWithId(other->nodes, other->numNodes, i);

        if (a == NULL && b == NULL)
        {
            cout << "Both nodes are NULL, this should not be possible" << endl;
            throw "Both nodes are NULL";
        }
        else if (a == NULL)
        {
            // use node b
        }
        else if (b == NULL)
        {
            // use node a
        }
        else
        {
            // both nodes exist

            // check if the nodes are matching i.e. have the same connections
        }
    }

    // 2. for each matching gene, randomly choose the gene from one of the parents
    // 3. for each disjoint gene, add the gene from the fittest parent
    // 4. for each excess gene, add the gene from the fittest parent

    throw "Not implemented";
}

int Genome::FeedForward(float *input_image)
{
    /**
     * 'Feed forward' algorithm implemented as a recursive search starting from the output nodes.
     *
     * @param input_image: Array of input values
     */
    // Reset all nodes to not ready
    for (int i = 0; i < numNodes; i++)
    {
        nodes[i]->Reset();
    }
    // Set input values
    for (int i = 0; i < numInputs; i++)
    {
        this->inputs[i]->LoadValue(input_image[i]);
    }
    // Run feed forward
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i]->CalculateValue();
    }
    // Find the output node with the highest value
    int max_index = 0;           // Index of the output node with the highest value
    float max_value = -INFINITY; // Value of the output node with the highest value
    for (int i = 0; i < numOutputs; i++)
    {
        if (outputs[i]->GetValue() > max_value)
        {
            max_value = outputs[i]->GetValue();
            max_index = i;
        }
    }

    return max_index;
}

void Genome::PrintInfo()
{
    /**
     * Print the information of the genome.
     */
    cout << "Genome name: " << name << endl;
    cout << "Genome fitness: " << fitness << endl;
    cout << "Genome numNodes: " << numNodes << endl;
    cout << "Genome numInputs: " << numInputs << endl;
    cout << "Genome numOutputs: " << numOutputs << endl;
    cout << "Genome numHidden: " << numHidden << endl;
}

// DEBUG
bool doesGenomeHaveConnections(Genome *genome)
{
    for (int i = 0; i < genome->GetNumNodes(); i++)
    {
        list<tuple<Node *, float, bool>> *connections = genome->GetNode(i)->GetConnections();
        if (connections->size() > 0)
        {
            return true;
        }
    }
    return false;
}

void Genome::PrintOutputLayerValues()
{
    for (int i = 0; i < numOutputs; i++)
    {
        cout << "Output " << i << " value: " << outputs[i]->GetValue() << endl;
    }
}

// Expose the Genome class to C
extern "C"
{
    Genome *InitGenome(char *name)
    {
        /**
         * Initialize a genome.
         *
         * @param name: Name of the genome
         *
         * @return: Pointer to the new genome
         */
        return new Genome(new string(name));
    }
    void DeleteGenome(Genome *genome)
    {
        /**
         * Delete a genome.
         *
         * @param genome: Pointer to the genome to delete
         */
        delete genome;
    }
    Genome *CopyGenome(Genome *genome)
    {
        /**
         * Copy a genome.
         *
         * @param genome: Pointer to the genome to copy
         *
         * @return: Pointer to the new genome
         */
        return new Genome(genome);
    }
    void CreateGenome(Genome *genome, int numInputs, int numOutputs)
    {
        /**
         * Create a base genome with a given number of input and output nodes, and connections from each output node to the bias node.
         *
         * @param genome: Pointer to the genome to initialize
         * @param numInputs: Number of input nodes
         * @param numOutputs: Number of output nodes
         */
        genome->InitGenome(numInputs, numOutputs);
    }
    void LoadGenome(Genome *genome, char *filename)
    {
        /**
         * Load a genome from a file.
         *
         * @param genome: Pointer to the genome to load
         * @param filename: Name of the file to load the genome from
         */
        genome->Load(new string(filename));
    }
    void SaveGenome(Genome *genome, char *filename)
    {
        /**
         * Save a genome to a file.
         *
         * @param genome: Pointer to the genome to save
         * @param filename: Name of the file to save the genome to
         */
        genome->Save(new string(filename));
    }
    void MutateGenome(Genome *genome)
    {
        /**
         * Mutate the genome.
         *
         * @param genome: Pointer to the genome to mutate
         */
        genome->Mutate();
    }
    Genome *CrossoverGenome(Genome *genome, Genome *other)
    {
        /**
         * Crossover the genome with another genome.
         *
         * @param genome: Pointer to the genome to crossover
         * @param other: Pointer to the other genome to crossover with
         */
        //  TODO: implement crossover algorithm
        // return genome->Crossover(other);
        return new Genome(genome);
    }
    int FeedForwardGenome(Genome *genome, float *input_image)
    {
        /**
         * 'Feed forward' algorithm implemented as a recursive search starting from the output nodes.
         *
         * @param genome: Pointer to the genome to feed forward
         * @param input_image: Array of input values
         *
         * @return: Index of the output node with the highest value
         */
        return genome->FeedForward(input_image);
    }
    void PrintGenomeInfo(Genome *genome)
    {
        /**
         * Print the information of the genome.
         *
         * @param genome: Pointer to the genome to print the information of
         */
        genome->PrintInfo();
    }
    void SetName(Genome *genome, char *name)
    {
        /**
         * Set the name of the genome.
         *
         * @param genome: Pointer to the genome to set the name of
         * @param name: Name of the genome
         */
        genome->SetName(new string(name));
    }
    const char *GetName(Genome *genome)
    {
        /**
         * Get the name of the genome.
         *
         * @param genome: Pointer to the genome to get the name of
         *
         * @return: Name of the genome
         */
        return genome->GetName()->c_str();
    }
    void SetFitness(Genome *genome, float fitness)
    {
        /**
         * Set the fitness of the genome.
         *
         * @param genome: Pointer to the genome to set the fitness of
         * @param fitness: Fitness of the genome
         */
        genome->SetFitness(fitness);
    }
    float GetFitness(Genome *genome)
    {
        /**
         * Get the fitness of the genome.
         *
         * @param genome: Pointer to the genome to get the fitness of
         *
         * @return: Fitness of the genome
         */
        return genome->GetFitness();
    }

    // DEBUG
    bool DoesGenomeHaveConnections(Genome *genome)
    {
        /**
         * Check if the genome has any connections.
         *
         * @param genome: Pointer to the genome to check
         *
         * @return: True if the genome has any connections, False otherwise
         */
        return doesGenomeHaveConnections(genome);
    }
    void PrintOutputLayerValues(Genome *genome)
    {
        /**
         * Print the values of the output layer of the genome.
         *
         * @param genome: Pointer to the genome to print the output layer values of
         */
        genome->PrintOutputLayerValues();
    }
}
