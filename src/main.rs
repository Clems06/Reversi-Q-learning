#![allow(non_snake_case)]
use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;
use serde_json::{Number, Value};


use std::io;
pub fn get_input(prompt: &str) -> String{
    println!("{}",prompt);
    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_goes_into_input_above) => {},
        Err(_no_updates_is_fine) => {},
    }
    input.trim().to_string()
}

pub struct Morpion {
    joueur: bool,
    grid: [Option<bool>; 9],
    finished: bool
}

impl Default for Morpion {
    fn default() -> Morpion {
        Morpion {
            joueur: false,
            grid: [None; 9],
            finished: false
        }
    }
}

impl Morpion {
    pub fn print_grid(&mut self) {
        println!("Game state:");
        for i in 0..3 {
            println!("{:?}", &self.grid[i*3..i*3+3]);
        }
    }

    pub fn check_valid_move(&mut self, spot: &usize) -> bool{
        self.grid[*spot].is_none()
    }

    pub fn get_valid_moves(&mut self) -> Vec<usize> {
        self.grid
            .iter()
            .enumerate()
            .filter_map(|(index, &r)| (r.is_none()).then(|| index))
            .collect::<Vec<_>>()
    }

    pub fn make_move(&mut self, value: &usize) -> f32 {
        self.grid[*value] = Some(self.joueur);
        if self.check_win(){
            self.finished = true;
            if self.joueur{return 100_f32;}
            return -100_f32;
        }
        if self.get_valid_moves().len() == 0{
            self.finished = true;
        }
        self.joueur = !self.joueur;
        0_f32
    }

    pub fn check_win(&mut self) -> bool{
        for i in 0..3{
            if self.grid[i*3..i*3+3].iter().all(|x| *x==Some(self.joueur)) ||
                self.grid[i..7+i].iter().step_by(3).all(|x| *x==Some(self.joueur)) {
                return true
            }
        };
        if self.grid[0]==Some(self.joueur) && self.grid[4]==Some(self.joueur) && self.grid[8]==Some(self.joueur){return true};
        if self.grid[2]==Some(self.joueur) && self.grid[4]==Some(self.joueur) && self.grid[6]==Some(self.joueur){return true};
        false
    }

    pub fn reset(&mut self) {
        self.joueur = false;
        self.grid = [None; 9];
        self.finished = false;
    }

}


pub struct BestState {
    spot: usize,
    qvalue: f32,
}

pub struct SmartMorpion {
    game: Morpion,
    qvalues: HashMap<i32, f32>,
    epsilon: f32,
    learning_rate: f32,
    min_exploration_proba: f32,
    exploration_decreasing_decay: f32,
    gamma: f32
}

impl Default for SmartMorpion {
    fn default() -> SmartMorpion {
        SmartMorpion {
            game: Default::default(),
            qvalues: HashMap::with_capacity(1000),
            epsilon: 1.0,
            min_exploration_proba: 0.1,
            exploration_decreasing_decay: 0.001,
            learning_rate: 0.1,
            gamma: 0.99
        }
    }
}

impl SmartMorpion {
    pub fn get_state(grid: &[Option<bool>; 9]) -> i32{
        let mut state: i32 = 0;
        for i in 0..9 {
            if grid[i].is_some() {
                if grid[i] == Some(true) {
                    state += 3_i32.pow(i.try_into().unwrap())
                } else {
                    state += 2*3_i32.pow(i.try_into().unwrap())
                }
            }
        };
        state
    }

    pub fn get_self_state(&mut self) -> i32 { SmartMorpion::get_state(&self.game.grid) }



    pub fn choose_action(&mut self) -> BestState{


        let possibilities: Vec<usize> = self.game.get_valid_moves();
        let mut future_grid: [Option<bool>; 9];
        let mut future_state: i32;
        let inequality_modifier:f32 = if self.game.joueur {1_f32} else {-1_f32};
        let mut best: BestState = BestState {spot: 0, qvalue: -1000.0*inequality_modifier};

        for spot in possibilities {
            future_grid = self.game.grid;
            future_grid[spot] = Some(self.game.joueur);
            future_state = SmartMorpion::get_state(&future_grid);

            if self.epsilon == 0_f32 {
                println!("Hash: {} Grid: {:?}, Qvalue: {}",future_state, future_grid, if self.qvalues.contains_key(&future_state) {self.qvalues[&future_state]} else {0.0});
            }
            //println!("Hash: {} Grid: {:?}, Qvalue: {}",future_state, future_grid, if self.qvalues.contains_key(&future_state) {self.qvalues[&future_state]} else {0.0});

            if (self.qvalues.contains_key(&future_state) && self.qvalues[&future_state]*inequality_modifier > best.qvalue*inequality_modifier) || 0_f32 > best.qvalue*inequality_modifier {
                best.spot = spot;
                best.qvalue = if self.qvalues.contains_key(&future_state) {self.qvalues[&future_state]} else { 0_f32 };
            }
        };
        best

    }

    pub fn main_loop(&mut self, n_episodes: i32){
        let max_iter_episode = 11;
        let mut rng = rand::thread_rng();
        let mut chosen: usize;

        for i in 0..n_episodes{

            for _y in 0..max_iter_episode{
                if rng.gen::<f32>() < self.epsilon {
                    let moves = self.game.get_valid_moves();
                    chosen = *moves.choose(&mut rng).unwrap();
                } else {
                    chosen = self.choose_action().spot;
                }

                //println!("Valid moves: {:?} Move chosen: {}", self.game.get_valid_moves(), chosen);
                let reward = self.game.make_move(&chosen);



                let new_state = self.get_self_state();
                //println!("New state: {} Reward: {}", new_state, reward);

                let added = if self.game.finished {reward} else {self.learning_rate*(reward + self.gamma*self.choose_action().qvalue)};
                //println!("{}", added);
                //println!();
                if self.qvalues.contains_key(&new_state){
                    *self.qvalues.get_mut(&new_state).unwrap() = (1_f32-self.learning_rate) * self.qvalues[&new_state] + added;
                } else {
                    self.qvalues.insert(new_state, added);
                }

                if self.game.finished {break;}

            }

            if self.epsilon != 0_f32 {
                self.epsilon = f32::max(self.min_exploration_proba, (-self.exploration_decreasing_decay*(i as f32)).exp());
            }

            self.game.reset();
            
        }
    }


}

fn main() {
    let load_progress: bool = true;
    let save_progress: bool = false;

    let learning_rate: f32 = 0.3;

    let mut smart: SmartMorpion;
    if load_progress{
        let loaded_grid = {
            // Load the first file into a string.
            let text = std::fs::read_to_string("morpion.json").unwrap();

            // Parse the string into a dynamically-typed JSON structure.
            serde_json::from_str::<HashMap<i32, f32>>(&text).unwrap()
        };

        smart = SmartMorpion {learning_rate: learning_rate, qvalues: loaded_grid, ..Default::default()};
        smart.main_loop(0);
        println!("{:?}", smart.qvalues);
    }
    else {
        smart = SmartMorpion { learning_rate: learning_rate, ..Default::default() };
        smart.main_loop(1000000);
        println!("{:?}", smart.qvalues);
    }


    /*let my_moves: [usize; 4] = [7_usize, 6_usize, 2_usize, 5_usize];
    let mut ai_move: usize;
    for my_move in my_moves{
        smart.game.make_move(&my_move);
        smart.game.print_grid();
        ai_move = smart.choose_action().spot;
        smart.game.make_move(&ai_move);
        smart.game.print_grid();
    }*/
    smart.epsilon = 0_f32;
    let mut my_move: usize;
    let mut ai_move: usize;



    if save_progress{
        std::fs::write(
            "morpion.json",
            serde_json::to_string_pretty(&smart.qvalues).unwrap(),
        ).unwrap();
    }
    for _i in 0..5 {
        my_move = get_input("Your move:").parse::<usize>().unwrap();
        smart.game.make_move(&my_move);
        smart.game.print_grid();
        ai_move = smart.choose_action().spot;
        smart.game.make_move(&ai_move);
        smart.game.print_grid();
    }

}
